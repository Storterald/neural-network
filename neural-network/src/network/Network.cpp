#include "Network.h"

#include <fstream>
#include <future>

#include "Layer.h"

constexpr float CLIP_EPSILON = 0.2f;
constexpr float GAMMA        = 0.99f;
constexpr float LAMBDA       = 0.95f;

Network::Network(
        uint32_t inputSize,
        uint32_t layerCount,
        const LayerCreateInfo layerInfos[],
        const std::filesystem::path &path) :

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
        uint32_t sampleCount,
        const float inputs[],
        const float outputs[]) {

#ifdef DEBUG_MODE_ENABLED
        // Since DEBUG mode uses the Logger, the program must
        // run in single thread to avoid gibberish in the log file.
        constexpr uint32_t threads = 1;
#else
        // If there are less samples than threads allocate
        // sampleCount threads.
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

                futures.emplace_back(_get_supervised_future(start, end, inputs, outputs));
        }

        // Wait for all threads to finish
        for (std::future<void> &f : futures)
                f.get();
}

void Network::encode(const std::filesystem::path &path) const
{
        // The file must be open in binary mode, and all
        // encode function must write binary.
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw LOGGER_EX("Error opening file.");

        // Calls the encode function on all layers, the encode
        // function uses std::ofstream::write, writing binary and
        // moving the position of std::ofstream::tellp.
        for (uint32_t L = 0; L < m_layerCount - 1; ++L)
                m_L[L]->encode(file);

        file.close();
}

std::unique_ptr<ILayer> *Network::_create_layers(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos) const {

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
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos,
        const char *path) const {

        if (layerCount == 0)
                throw LOGGER_EX("Cannot initialize Network with no layers. The "
                                "minimum required amount is 1, the output layer.");

        // If the path is empty use other layer constructor.
        if (path[0] == '\0')
                return _create_layers(layerCount, layerInfos);

        if (!std::filesystem::exists(path))
                throw LOGGER_EX("File given to Network() does not exist.");

        std::ifstream file(path, std::ios::binary);
        auto layers = new std::unique_ptr<ILayer>[layerCount];
        layers[0]   = Layer::create(m_inputSize, layerInfos[0], file);

        for (uint32_t L = 1; L < layerCount; ++L)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L], file);

        file.close();
        return layers;
}

uint32_t *Network::_get_sizes(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos) const {

        auto sizes = new uint32_t[layerCount];
        sizes[0]   = m_inputSize;

        for (uint32_t L = 1; L < layerCount; ++L)
                sizes[L] = layerInfos[L - 1].neuronCount;

        return sizes;
}

std::future<void> Network::_get_supervised_future(
        uint32_t start,
        uint32_t end,
        const float inputs[],
        const float outputs[]) {

        return std::async(std::launch::async, [&]() -> void {
                for (uint32_t i = start; i < end; ++i) {
                        // All activation values are stored to back-propagate the network.
                        const auto a = new Vector[m_layerCount];
                        a[0]         = Vector(m_inputSize, inputs);

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                a[L] = m_L[L - 1]->forward(a[L - 1]);

                        // The cost of the last layer neurons is calculated with
                        // (ajL - yj) ^ 2, this mean that the derivative ∂C is
                        // equal to 2 * (ajL - y).
                        Vector dC = (a[m_layerCount - 1] - Vector(m_outputSize, outputs + i * m_outputSize)) * 2.0f;

                        this->backward(dC, a);
                        delete [] a;
                }
        });
}

void Network::_train_ppo(
        Network &valueNetwork,
        IEnvironment &environment,
        uint32_t epochs,
        uint32_t maxSteps) {

        struct IterationData {
                uint64_t        stateHash;
                float           reward;
                float           predictedReward;
                Vector          policy;
                Vector          *policyActivations;
                Vector          *valueActivations;
        };

        // Saving variables here for quicker and clearer access
        const uint32_t valueLayerCount = valueNetwork.m_layerCount;

        // Keep policies between epochs.
        std::unordered_map<uint64_t, Vector> oldPolicies;

        for (uint32_t i = 0; i < epochs; ++i) {
                std::vector<IterationData> iterationData;
                for (uint32_t s = 0, done = false; !done && s < maxSteps; ++s) {
                        iterationData.push_back({});
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData.back();

                        // The current state of the environment, updated
                        // every time IEnvironment::step() is called.
                        const Vector state = environment.getState();

                        // Save the state hash only since the values won't be used.
                        stateHash = state.hash();

                        // Forward function of the policy network, keeping
                        // activation values.
                        pa    = new Vector[m_layerCount];
                        pa[0] = state;

                        for (uint32_t L = 1; L < m_layerCount; ++L)
                                pa[L] = m_L[L - 1]->forward(pa[L - 1]);

                        // Saving the chosen policy in a different vector
                        // as it will substitute the oldPolicies reference.
                        policy = pa[m_layerCount - 1];

                        // Forward function of the value network, keeping
                        // activation values. Value are not used right now,
                        // but they will be used later.
                        va    = new Vector[valueLayerCount];
                        va[0] = state;

                        for (uint32_t L = 1; L < valueLayerCount; ++L)
                                va[L] = valueNetwork.m_L[L - 1]->forward(va[L - 1]);

                        // Saving the predicted rewards in a different vector
                        // to make the next steps easier.
                        predictedReward = va[valueLayerCount - 1].at(0);

                        // Step the Environment forward, updating the state.
                        auto [_reward, _done] = environment.step(pa[m_layerCount - 1]);
                        reward                = _reward;
                        done                  = _done;
                }

#ifdef DEBUG_MODE_ENABLED
                Logger::Log() << LOGGER_PREF(LOG_DEBUG) << "Execution [" << i << "] done.\n";
#endif // DEBUG_MODE_ENABLED

                const auto STATE_COUNT = (uint32_t)iterationData.size();
                float *advantages      = new float[STATE_COUNT], previous = 0.0f;

                // Compute advantages before the training as the calculation is done
                // backwards to use the previous advantage as a part of the formula.
                for (int32_t s = (int32_t)STATE_COUNT - 1; s >= 0; --s) {
                        const float reward              = iterationData[s].reward;
                        const float predictedReward     = iterationData[s].predictedReward;
                        const float nextPredictedReward = (uint32_t)s == STATE_COUNT - 1 ?
                                0 : iterationData[s + 1].predictedReward;

                        const float delta = reward + GAMMA * nextPredictedReward * predictedReward;
                        advantages[s]     = delta + GAMMA * LAMBDA * previous;
                        previous          = advantages[s];
                }

                // Trains the networks
                for (uint32_t s = 0; s < STATE_COUNT; ++s) {
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData[s];

                        // Save content of arrays in variables for quicker access.
                        const float advantage = advantages[s];
                        policy = policy.max(EPSILON);

                        // Value network cost, the cost in a PPO environment is a single
                        // value back-propagated throughout all the output neurons.
                        Vector vdC = { (predictedReward - reward) * 2.0f };

                        // Train the value network and delete dynamically allocated array.
                        valueNetwork.backward(vdC, va);
                        delete [] va;

                        // If the current state has never been met, the old policy
                        // is set to the current policy.
                        oldPolicies.insert({stateHash, policy});

                        //               ⌈      π𝜃ɴᴇᴡ(aₜ|sₜ)                           ⌉
                        // Lᴘᴏʟɪᴄʏ(𝜃) = 𝔼| min( ⎯⎯⎯⎯⎯⎯⎯⎯ Aₜ, clip(p, 1 - ε, 1 + ε)Aₜ) |
                        //               ⌊      π𝜃ᴏʟᴅ(aₜ|sₜ)                           ⌋
                        const Vector ratio         = policy / oldPolicies.at(stateHash);
                        const Vector clippedRatio  = ratio.clamp(1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON);
                        const Vector surrogateLoss = (ratio * advantage).min(clippedRatio * advantage);

                        // Replace the old policy with the new one, it is important to
                        // save the policy after .max(EPSILON) is done to prevent division
                        // by 0 in the 'ratio' calculation.
                        oldPolicies[stateHash] = policy;

                        // Train the policy network and delete dynamically allocated array.
                        this->backward(surrogateLoss, pa);
                        delete [] pa;
                }

                // Delete advantages and reset environment in preparation for
                // next epoch.
                delete [] advantages;
                environment.reset();

#ifdef DEBUG_MODE_ENABLED
                Logger::Log() << LOGGER_PREF(LOG_DEBUG) << "Training [" << i << "] done.\n";
#endif // DEBUG_MODE_ENABLED
        }
}