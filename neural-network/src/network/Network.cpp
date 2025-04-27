#include <neural-network/network/Network.h>

#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <vector>
#include <future>

#include <neural-network/network/PPO/IEnvironment.h>
#include <neural-network/network/Layer.h>
#include <neural-network/types/Vector.h>
#include <neural-network/utils/Logger.h>
#include <neural-network/Base.h>

NN_BEGIN

Network::Network(
        uint32_t                           inputSize,
        uint32_t                           layerCount,
        const LayerCreateInfo              layerInfos[],
        const std::filesystem::path        &path) :

        m_layerCount(layerCount + 1),
        m_inputSize(inputSize),
        m_outputSize(layerInfos[layerCount - 1].neuronCount),
        m_L(_create_layers(layerCount, layerInfos, path.string().c_str())),
        m_n(_get_sizes(layerCount, layerInfos)) {}

Network::~Network()
{
        delete[] m_L;
        delete[] m_n;
}

Vector Network::forward(Vector aL) const
{
        // Not passing as const ref to avoid creating an extra Vector to store
        // the current activation values.

        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                aL = m_L[L]->forward(aL);

        return aL;
}

void Network::backward(const Vector &input, const Vector &dC)
{
        auto a = new Vector[m_layerCount];
        a[0]   = input;

        for (uint32_t L = 1; L < m_layerCount; ++L)
                a[L] = m_L[L - 1]->forward(a[L - 1]);

        this->backward(dC, a);

        delete [] a;
}


void Network::backward(Vector dC, const Vector a[])
{
        for (int32_t L = (int32_t)m_layerCount - 2; L >= 0; --L)
                dC = m_L[L]->backward(dC, a[L]);
}

void Network::train_supervised(
        uint32_t           sampleCount,
        const float        inputs[],
        const float        outputs[]) {

#ifdef DEBUG_MODE_ENABLED
        constexpr uint32_t threads = 1;
#else
        const uint32_t threads = std::min(std::thread::hardware_concurrency(), sampleCount);
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
                offset              += count;

                futures.push_back(_get_supervised_future(
                        start, end, inputs, outputs));
        }

        for (std::future<void> &f : futures)
                f.get();
}

void Network::encode(const std::filesystem::path &path) const
{
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw LOGGER_EX("Error opening file.");

        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                m_L[L]->encode(file);

        file.close();
}

std::unique_ptr<ILayer> *Network::_create_layers(
        uint32_t                     layerCount,
        const LayerCreateInfo        *layerInfos) const {

        if (layerCount == 0)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        auto layers = new std::unique_ptr<ILayer>[layerCount];
        layers[0]   = Layer::create(m_inputSize, layerInfos[0]);

        for (uint32_t L = 1; L < layerCount; ++L)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L]);

        return layers;
}

std::unique_ptr<ILayer> *Network::_create_layers(
        uint32_t                     layerCount,
        const LayerCreateInfo        *layerInfos,
        const std::string_view       &path) const {

        namespace fs = std::filesystem;

        if (layerCount == 0)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        if (path.empty())
                return _create_layers(layerCount, layerInfos);

        if (!fs::exists(path))
                throw LOGGER_EX("File given to Network() does not exist.");

        std::ifstream file(fs::path(path), std::ios::binary);
        auto layers = new std::unique_ptr<ILayer>[layerCount];
        layers[0]   = Layer::create(m_inputSize, layerInfos[0], file);

        for (uint32_t L = 1; L < layerCount; ++L)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L], file);

        file.close();
        return layers;
}

uint32_t *Network::_get_sizes(
        uint32_t                     layerCount,
        const LayerCreateInfo        *layerInfos) const {

        auto sizes = new uint32_t[layerCount];
        sizes[0]   = m_inputSize;

        for (uint32_t L = 1; L < layerCount; ++L)
                sizes[L] = layerInfos[L - 1].neuronCount;

        return sizes;
}

std::future<void> Network::_get_supervised_future(
        uint32_t           start,
        uint32_t           end,
        const float        inputs[],
        const float        outputs[]) {

        return std::async(std::launch::async, [&]() -> void {
                for (uint32_t i = start; i < end; ++i) {
                        const auto a = new Vector[m_layerCount];
                        a[0]         = Vector(m_inputSize, inputs);

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                a[L] = m_L[L - 1]->forward(a[L - 1]);

                        const Vector y(m_outputSize, outputs + (uintptr_t)i * m_outputSize);
                        const Vector dC = 2.0f * (a[m_layerCount - 1] - y);

                        this->backward(dC, a);
                        delete [] a;
                }
        });
}

void Network::_train_ppo(
        Network             &valueNetwork,
        IEnvironment        &environment,
        uint32_t            epochs,
        uint32_t            maxSteps) {

        struct IterationData {
                uint64_t        stateHash;
                float           reward;
                float           predictedReward;
                Vector          policy;
                Vector          *policyActivations;
                Vector          *valueActivations;
        };

        const uint32_t valueLayerCount = valueNetwork.m_layerCount;

        // The policies are saved between epochs for better training.
        std::unordered_map<uint64_t, Vector> oldPolicies;

        for (uint32_t i = 0; i < epochs; ++i) {
                std::vector<IterationData> iterationData;
                for (uint32_t s = 0, done = false; !done && s < maxSteps; ++s) {
                        iterationData.emplace_back();
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData.back();

                        const Vector state = environment.getState();
                        stateHash = std::hash<Data>{}(state);

                        pa    = new Vector[m_layerCount];
                        pa[0] = state;

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                pa[L] = m_L[L - 1]->forward(pa[L - 1]);

                        policy = pa[m_layerCount - 1];

                        va    = new Vector[valueLayerCount];
                        va[0] = state;

                        for (uint32_t L = 1; L < valueLayerCount; ++L)
                                va[L] = valueNetwork.m_L[L - 1]->forward(va[L - 1]);

                        predictedReward = va[valueLayerCount - 1].at(0);

                        auto [_reward, _done] = environment.step(pa[m_layerCount - 1]);
                        reward                = _reward;
                        done                  = _done;
                }

#ifdef DEBUG_MODE_ENABLED
                Logger::Log() << LOGGER_PREF(DEBUG) << "Execution [" << i << "] done.\n";
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
                        const Vector vdC      = { (predictedReward - reward) * 2.0f };

                        valueNetwork.backward(vdC, va);
                        delete [] va;

                        oldPolicies.insert({stateHash, policy});

                        //               ⌈      π𝜃ɴᴇᴡ(aₜ|sₜ)                           ⌉
                        // Lᴘᴏʟɪᴄʏ(𝜃) = 𝔼| min( ⎯⎯⎯⎯⎯⎯⎯⎯ Aₜ, clip(p, 1 - ε, 1 + ε)Aₜ) |
                        //               ⌊      π𝜃ᴏʟᴅ(aₜ|sₜ)                           ⌋
                        const Vector ratio         = policy / oldPolicies.at(stateHash);
                        const Vector clippedRatio  = ratio.clamp(1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON);
                        const Vector surrogateLoss = (ratio * advantage).min(clippedRatio * advantage);

                        oldPolicies[stateHash] = policy;

                        this->backward(surrogateLoss, pa);
                        delete [] pa;
                }

                delete [] advantages;
                environment.reset();

#ifdef DEBUG_MODE_ENABLED
                Logger::Log() << LOGGER_PREF(DEBUG) << "Training [" << i << "] completed.\n";
#endif // DEBUG_MODE_ENABLED
        }
}

NN_END
