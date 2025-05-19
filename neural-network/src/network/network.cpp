#include <neural-network/network/network.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <driver_types.h> // cudaStream_t

#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

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
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>
#include <neural-network/base.h>

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

#ifdef BUILD_CUDA_SUPPORT
        m_stream(_create_stream()),
#endif // BUILD_CUDA_SUPPORT
        m_layerCount((uint32_t)std::distance(infosBegin, infosEnd) + 1),
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

vector network::forward(vector aL) const
{
        // Not passing as const ref to avoid creating an extra vector to store
        // the current activation values.

        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                aL = m_L[L]->forward(aL);

#ifdef BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaStreamSynchronize(m_stream),
                "Error synchronizing in network::backward.");
#endif // BUILD_CUDA_SUPPORT

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

#ifdef BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaStreamSynchronize(m_stream),
                "Error synchronizing in network::backward.");
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

        auto inputsChunks = inputs  | std::views::chunk(chunks);
        auto outputsChunk = outputs | std::views::chunk(chunks);

        for (uint32_t t = 0; t < chunks; ++t) {
                const size_t size     = inputsChunks[t].size();
                const float *cInputs  = inputsChunks[t].data();
                const float *cOutputs = outputsChunk[t].data();

                futures.push_back(_get_supervised_future((uint32_t)size, cInputs, cOutputs));
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

#ifdef BUILD_CUDA_SUPPORT
cudaStream_t network::_create_stream() const
{
        static std::array<std::optional<cudaStream_t>, 16> streams{};
        static uint32_t usedStreams = 0;

        if (!streams[usedStreams].has_value()) {
                cudaStream_t stream;
                CUDA_CHECK_ERROR(cudaStreamCreate(&stream),
                         "Could not create CUDA stream.");
                streams[usedStreams] = stream;
        }

        const cudaStream_t stream = streams[usedStreams].value();
        usedStreams               = (usedStreams + 1) % streams.size();
        return stream;
}
#endif // BUILD_CUDA_SUPPORT

std::unique_ptr<layer> *network::_create_layers(
        const layer_create_info        *infos) const {

        if (m_layerCount == 1)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        const uint32_t size = m_layerCount - 1;
        bool useCuda        = false;

#ifdef BUILD_CUDA_SUPPORT
        const auto pred = [](uint32_t count, const layer_create_info &value) {
                return count + value.neuronCount;
        };
        const uint32_t neurons = std::accumulate(infos, infos + size, m_inputSize, pred);
        useCuda                = neurons >= CUDA_MINIMUM;
#endif // BUILD_CUDA_SUPPORT

        auto layers = new std::unique_ptr<layer>[size];
        if (!useCuda)
                ::_create_layers(layers, m_inputSize, size, infos);
#ifdef BUILD_CUDA_SUPPORT
        else
                ::_create_layers(layers, m_inputSize, size, infos, m_stream);
#endif // BUILD_CUDA_SUPPORT

        return layers;
}

std::unique_ptr<layer> *network::_create_layers(
        const layer_create_info        *infos,
        const std::string_view         &path) const {

        namespace fs = std::filesystem;

        if (path.empty())
                return _create_layers(infos);

        if (m_layerCount == 1)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        if (!fs::exists(path))
                throw LOGGER_EX("File given to Network() does not exist.");

        const uint32_t size = m_layerCount - 1;
        bool useCuda        = false;

#ifdef BUILD_CUDA_SUPPORT
        const auto pred = [](uint32_t count, const layer_create_info &value) {
                return count + value.neuronCount;
        };
        const uint32_t neurons = std::accumulate(infos, infos + size, m_inputSize, pred);
        useCuda                = neurons >= CUDA_MINIMUM;
#endif // BUILD_CUDA_SUPPORT

        std::ifstream file(fs::path(path), std::ios::binary);
        auto layers = new std::unique_ptr<layer>[size];
        if (!useCuda)
                ::_create_layers(layers, m_inputSize, size, infos, file);
#ifdef BUILD_CUDA_SUPPORT
        else
                ::_create_layers(layers, m_inputSize, size, infos, file, m_stream);
#endif // BUILD_CUDA_SUPPORT

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
        uint32_t           sampleSize,
        const float        inputs[],
        const float        outputs[]) {

        namespace ch = std::chrono;

        return std::async(std::launch::async, [&]() -> void {
                auto s = ch::high_resolution_clock::now();

                for (uint32_t i = 0; i < sampleSize; ++i) {
                        const float *inValues  = inputs + (uintptr_t)i * m_inputSize;
                        const float *outValues = outputs + (uintptr_t)i * m_outputSize;

                        const auto a = new vector[m_layerCount];
                        a[0]         = vector(m_inputSize, inValues);

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                a[L] = m_L[L - 1]->forward(a[L - 1]);

                        const vector y(m_outputSize, outValues);
                        const vector dC = 2.0f * (a[m_layerCount - 1] - y);

                        this->backward(dC, a);
                        delete [] a;
                }

                auto e = ch::high_resolution_clock::now();
                auto time = ch::duration_cast<ch::milliseconds>(e - s);
                logger::log() << "Thread [" << std::this_thread::get_id()
                              << "] finished execution in " << time << "\n";
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
