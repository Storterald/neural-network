#include <neural-network/network/network.h>

#include <unordered_map>
#include <filesystem>
#include <optional>
#include <numeric> // std::accumulate
#include <fstream>
#include <cstdint>
#include <chrono>
#include <vector>
#include <thread>
#include <future>

#include <neural-network/network/PPO/environment.h>
#include <neural-network/utils/exceptions.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>
#include <neural-network/base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

template<typename ...Args>
static void _create_layers(
        std::unique_ptr<nn::layer>         *layers,
        uint32_t                           inputSize,
        uint32_t                           count,
        const nn::layer_create_info        *infos,
        Args                               &&...args) {

        layers[0] = nn::layer::create(inputSize, infos[0], args...);
        for (uint32_t L = 1; L < count; ++L)
                layers[L] = nn::layer::create(infos[L - 1].neuronCount, infos[L], args...);
}

namespace nn {

network::network(
        uint32_t                           inputSize,
        const layer_create_info            *infosBegin,
        const layer_create_info            *infosEnd,
        const std::filesystem::path        &path) :

        m_layerCount((uint32_t)std::distance(infosBegin, infosEnd) + 1),
#ifdef BUILD_CUDA_SUPPORT
        m_stream(_create_stream(infosBegin)),
#endif // BUILD_CUDA_SUPPORT
        m_inputSize(inputSize),
        m_outputSize(infosBegin[m_layerCount - 2].neuronCount),
        m_L(_create_layers(infosBegin, path.string().c_str())),
        m_n(_get_sizes(infosBegin)) {
}

network::~network()
{
        delete[] m_L;
        delete[] m_n;
}

vector network::forward(const vector &input) const
{
        const vector out = _forward_impl(input);

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT

        return out;
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

void network::backward(const vector &cost, const vector activationValues[])
{
        _backward_impl(cost, activationValues);

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT
}

void network::train_supervised(
        uint32_t           samplesCount,
        const float        _inputs[],
        const float        _outputs[]) {

        namespace ch = std::chrono;

#ifdef DEBUG_MODE_ENABLED
        constexpr uint32_t chunks = 1;
#else
        const uint32_t threads = std::thread::hardware_concurrency();
        const uint32_t chunks  = std::clamp(threads, 1u, samplesCount);
#endif // DEBUG_MODE_ENABLED

        std::vector<std::future<void>> futures;
        futures.reserve(chunks);

        std::span inputs(_inputs, (size_t)samplesCount * m_inputSize);
        std::span outputs(_outputs, (size_t)samplesCount * m_outputSize);

        auto inputsChunks = inputs  | std::views::chunk(inputs.size() / chunks);
        auto outputsChunk = outputs | std::views::chunk(outputs.size() / chunks);

        for (uint32_t t = 0; t < chunks; ++t)
                futures.push_back(_get_supervised_future(inputsChunks[t], outputsChunk[t]));

        for (std::future<void> &f : futures)
                f.get();

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT
}

void network::encode(const std::filesystem::path &path) const
{
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw fatal_error("Error opening file.");

        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                m_L[L]->encode(file);

        file.close();
}

#ifdef BUILD_CUDA_SUPPORT
stream network::_create_stream(const layer_create_info *infos) const
{
        static std::array<stream, 16> streams = []() {
                std::array<stream, 16> ret{};
                for (stream &str : ret)
                        str = cuda::create_stream();

                return ret;
        }();
        static uint32_t usedStreams = 0;

        uint32_t neurons = m_inputSize * infos[0].neuronCount;
        for (uint32_t i = 1; i < m_layerCount - 1; ++i)
                neurons += infos[i - 1].neuronCount * infos[i].neuronCount;

        if (neurons >= CUDA_MINIMUM) {
                stream str  = streams[usedStreams];
                usedStreams = (usedStreams + 1) % streams.size();
                return str;
        }

        return invalid_stream;
}
#endif // BUILD_CUDA_SUPPORT

std::unique_ptr<layer> *network::_create_layers(
        const layer_create_info        *infos) const {

        if (m_layerCount == 1)
                throw fatal_error("Cannot initialize Network with no layers. The "
                                  "minimum required amount is 1, the output layer.");

        const uint32_t size = m_layerCount - 1;

        auto layers = new std::unique_ptr<layer>[size];
        ::_create_layers(layers, m_inputSize, size, infos, m_stream);

        return layers;
}

std::unique_ptr<layer> *network::_create_layers(
        const layer_create_info        *infos,
        const std::string_view         &path) const {

        namespace fs = std::filesystem;

        if (path.empty())
                return _create_layers(infos);

        if (m_layerCount == 1)
                throw fatal_error("Cannot initialize Network with no layers. The "
                                  "minimum required amount is 1, the output layer.");

        if (!fs::exists(path))
                throw fatal_error("File given to Network() does not exist.");

        const uint32_t size = m_layerCount - 1;

        std::ifstream file(fs::path(path), std::ios::binary);
        auto layers = new std::unique_ptr<layer>[size];
        ::_create_layers(layers, m_inputSize, size, infos, file, m_stream);

        file.close();
        return layers;
}

uint32_t *network::_get_sizes(
        const layer_create_info        *infos) const {

        const uint32_t size = m_layerCount - 1;

        auto sizes = new uint32_t[size];
        sizes[0]   = m_inputSize;

        for (uint32_t L = 1; L < size; ++L)
                sizes[L] = infos[L - 1].neuronCount;

        return sizes;
}

std::future<void> network::_get_supervised_future(
        const chunk        &inputs,
        const chunk        &outputs) {

        namespace ch = std::chrono;
        using clock  = ch::high_resolution_clock;

        return std::async(std::launch::async, [&]() -> void {
                const float *in   = inputs.data();
                const float *out  = outputs.data();
                const size_t size = inputs.size() / m_inputSize;

                const auto s = clock::now();
                for (uint32_t i = 0; i < size; ++i) {
                        const float *inValues  = in + (uintptr_t)i * m_inputSize;
                        const float *outValues = out + (uintptr_t)i * m_outputSize;

                        const auto a = new vector[m_layerCount];
                        a[0]         = vector(m_inputSize, inValues, m_stream);

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                a[L] = m_L[L - 1]->forward(a[L - 1]);

                        const vector y(m_outputSize, outValues);
                        const vector dC = 2.0f * (a[m_layerCount - 1] - y);

                        // TODO i'm not sure if this can be done with _backward_impl
                        this->backward(dC, a);
                        delete [] a;
                }

                const auto e    = clock::now();
                const auto time = ch::duration_cast<ch::milliseconds>(e - s);

                logger::log() << "Thread [" << std::this_thread::get_id()
                              << "] finished execution in " << time << "\n";
        });
}

vector network::_forward_impl(vector aL) const
{
        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                aL = m_L[L]->forward(aL);

        return aL;
}

void network::_backward_impl(vector dC, const vector a[])
{
        for (int32_t L = (int32_t)m_layerCount - 2; L >= 0; --L)
                dC = m_L[L]->backward(dC, a[L]);
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
                logger::log() << logger::pref(LOG_DEBUG) << "Execution [" << i << "] done.\n";
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
                logger::log() << logger::pref(LOG_DEBUG) << "Training [" << i << "] completed.\n";
#endif // DEBUG_MODE_ENABLED
        }
}

} // namespace nn
