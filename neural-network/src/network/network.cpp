#include <neural-network/network/network.h>

#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include <future>

#include <neural-network/network/PPO/environment.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>
#include <neural-network/base.h>

namespace nn {

network::network(
        uint32_t                           inputSize,
        uint32_t                           layerCount,
        const layer_create_info            layerInfos[],
        const std::filesystem::path        &path) :

        m_layerCount(layerCount + 1),
        m_inputSize(inputSize),
        m_outputSize(layerInfos[layerCount - 1].neuronCount),
        m_L(_create_layers(layerCount, layerInfos, path.string().c_str())),
        m_n(_get_sizes(layerCount, layerInfos)) {}

network::~network()
{
        delete[] m_L;
        delete[] m_n;
}

vector network::forward(vector aL) const
{
        // Not passing as const ref to avoid creating an extra vector to store
        // the current activation values.

        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                aL = m_L[L]->forward(aL);

        return aL;
}

void network::backward(const vector &input, const vector &dC)
{
        auto a = new vector[m_layerCount];
        a[0]   = input;

        for (uint32_t L = 1; L < m_layerCount; ++L)
                a[L] = m_L[L - 1]->forward(a[L - 1]);

        this->backward(dC, a);

        delete [] a;
}


void network::backward(vector dC, const vector a[])
{
        for (int32_t L = (int32_t)m_layerCount - 2; L >= 0; --L)
                dC = m_L[L]->backward(dC, a[L]);
}

void network::train_supervised(
        uint32_t           sampleCount,
        const float        inputs[],
        const float        outputs[]) {

#ifdef DEBUG_MODE_ENABLED
        constexpr uint32_t threads = 1;
#else
        const uint32_t threads = std::clamp(std::thread::hardware_concurrency(), 1u, sampleCount);
#endif // DEBUG_MODE_ENABLED

        std::vector<std::future<void>> futures;
        futures.reserve(threads);

        const uint32_t batchSize = sampleCount / threads;
        const uint32_t remainder = sampleCount % threads;
        uint32_t offset          = 0;

        for (uint32_t t = 0; t < threads; ++t) {
                const uint32_t count = batchSize + (t < remainder ? 1 : 0);
                const uint32_t start = offset;
                const uint32_t end   = start + count;
                offset               = end;

                futures.push_back(_get_supervised_future(
                        start, end, inputs, outputs));
        }

        for (std::future<void> &f : futures)
                f.get();
}

void network::encode(const std::filesystem::path &path) const
{
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw LOGGER_EX("Error opening file.");

        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                m_L[L]->encode(file);

        file.close();
}

std::unique_ptr<layer> *network::_create_layers(
        uint32_t                       layerCount,
        const layer_create_info        *layerInfos) const {

        if (layerCount == 0)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        auto layers = new std::unique_ptr<layer>[layerCount];
        layers[0]   = layer::create(m_inputSize, layerInfos[0]);

        for (uint32_t L = 1; L < layerCount; ++L)
                layers[L] = layer::create(layerInfos[L - 1].neuronCount, layerInfos[L]);

        return layers;
}

std::unique_ptr<layer> *network::_create_layers(
        uint32_t                       layerCount,
        const layer_create_info        *layerInfos,
        const std::string_view         &path) const {

        namespace fs = std::filesystem;

        if (layerCount == 0)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        if (path.empty())
                return _create_layers(layerCount, layerInfos);

        if (!fs::exists(path))
                throw LOGGER_EX("File given to Network() does not exist.");

        std::ifstream file(fs::path(path), std::ios::binary);
        auto layers = new std::unique_ptr<layer>[layerCount];
        layers[0]   = layer::create(m_inputSize, layerInfos[0], file);

        for (uint32_t L = 1; L < layerCount; ++L)
                layers[L] = layer::create(layerInfos[L - 1].neuronCount, layerInfos[L], file);

        file.close();
        return layers;
}

uint32_t *network::_get_sizes(
        uint32_t                       layerCount,
        const layer_create_info        *layerInfos) const {

        auto sizes = new uint32_t[layerCount];
        sizes[0]   = m_inputSize;

        for (uint32_t L = 1; L < layerCount; ++L)
                sizes[L] = layerInfos[L - 1].neuronCount;

        return sizes;
}

std::future<void> network::_get_supervised_future(
        uint32_t           start,
        uint32_t           end,
        const float        inputs[],
        const float        outputs[]) {

        namespace ch = std::chrono;

        return std::async(std::launch::async, [&]() -> void {
                auto s = ch::high_resolution_clock::now();
                for (uint32_t i = start; i < end; ++i) {
                        const auto a = new vector[m_layerCount];
                        a[0]         = vector(m_inputSize, inputs + (uintptr_t)i * m_inputSize);

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                a[L] = m_L[L - 1]->forward(a[L - 1]);

                        const vector y(m_outputSize, outputs + (uintptr_t)i * m_outputSize);
                        const vector dC = 2.0f * (a[m_layerCount - 1] - y);

                        this->backward(dC, a);
                        delete [] a;
                }
                auto e = ch::high_resolution_clock::now();
                logger::log() << "Thread [" << std::this_thread::get_id() << "] "
                        "finished execution in " << ch::duration_cast<ch::milliseconds>(e - s) << "\n";
        });
}

void network::_train_ppo(
        network            &valueNetwork,
        environment        &environment,
        uint32_t           epochs,
        uint32_t           maxSteps) {

        struct iteration_data {
                uint64_t        stateHash;
                float           reward;
                float           predictedReward;
                vector          policy;
                vector          *policyActivations;
                vector          *valueActivations;

        };

        const uint32_t valueLayerCount = valueNetwork.m_layerCount;

        // The policies are saved between epochs for better training.
        std::unordered_map<uint64_t, vector> oldPolicies;

        for (uint32_t i = 0; i < epochs; ++i) {
                std::vector<iteration_data> iterationData;
                for (uint32_t s = 0, done = false; !done && s < maxSteps; ++s) {
                        iterationData.emplace_back();
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData.back();

                        const vector state = environment.getState();
                        stateHash = std::hash<buf>{}(state);

                        pa    = new vector[m_layerCount];
                        pa[0] = state;

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                pa[L] = m_L[L - 1]->forward(pa[L - 1]);

                        policy = pa[m_layerCount - 1];

                        va    = new vector[valueLayerCount];
                        va[0] = state;

                        for (uint32_t L = 1; L < valueLayerCount; ++L)
                                va[L] = valueNetwork.m_L[L - 1]->forward(va[L - 1]);

                        predictedReward = va[valueLayerCount - 1].at(0);

                        auto [_reward, _done] = environment.step(pa[m_layerCount - 1]);
                        reward                = _reward;
                        done                  = _done;
                }

#ifdef DEBUG_MODE_ENABLED
                logger::log() << LOGGER_PREF(DEBUG) << "Execution [" << i << "] done.\n";
#endif // DEBUG_MODE_ENABLED

                const auto STATE_COUNT = (uint32_t)iterationData.size();
                float *advantages      = new float[STATE_COUNT], previous = 0.0f;

                for (int32_t s = (int32_t)STATE_COUNT - 1; s >= 0; --s) {
                        const float reward              = iterationData[s].reward;
                        const float predictedReward     = iterationData[s].predictedReward;
                        const float nextPredictedReward = (uint32_t)s == STATE_COUNT - 1 ?
                                0 : iterationData[s + 1].predictedReward;

                        const float delta = reward + GAMMA * nextPredictedReward * predictedReward;
                        advantages[s]     = delta + GAMMA * LAMBDA * previous;
                        previous          = advantages[s];
                }

                for (uint32_t s = 0; s < STATE_COUNT; ++s) {
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData[s];

                        const float advantage = advantages[s];
                        policy                = policy.max(EPSILON); // prevent division by 0.
                        const vector vdC      = { (predictedReward - reward) * 2.0f };

                        valueNetwork.backward(vdC, va);
                        delete [] va;

                        oldPolicies.insert({stateHash, policy});

                        //               ⌈      π𝜃ɴᴇᴡ(aₜ|sₜ)                           ⌉
                        // Lᴘᴏʟɪᴄʏ(𝜃) = 𝔼| min( ⎯⎯⎯⎯⎯⎯⎯⎯ Aₜ, clip(p, 1 - ε, 1 + ε)Aₜ) |
                        //               ⌊      π𝜃ᴏʟᴅ(aₜ|sₜ)                           ⌋
                        const vector ratio         = policy / oldPolicies.at(stateHash);
                        const vector clippedRatio  = ratio.clamp(1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON);
                        const vector surrogateLoss = (ratio * advantage).min(clippedRatio * advantage);

                        oldPolicies[stateHash] = policy;

                        this->backward(surrogateLoss, pa);
                        delete [] pa;
                }

                delete [] advantages;
                environment.reset();

#ifdef DEBUG_MODE_ENABLED
                logger::log() << LOGGER_PREF(DEBUG) << "Training [" << i << "] completed.\n";
#endif // DEBUG_MODE_ENABLED
        }
}

} // namespace nn
