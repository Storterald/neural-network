#include "Train.h"

#include <future>
#include <ranges>

#include "Base.h"

void Train::supervised(
        Network &network,
        uint32_t sampleCount,
        const float inputs[],
        const float outputs[]) {

#ifdef DEBUG_MODE_ENABLED
        // Since DEBUG mode uses the Logger, the program must
        // run in single thread to avoid gibberish in the log file.
        constexpr uint32_t THREAD_COUNT { 1 };
#else
        // If there are less samples than threads allocate
        // sampleCount threads.
        const uint32_t THREAD_COUNT { std::min(std::thread::hardware_concurrency(), sampleCount) };
#endif // DEBUG_MODE_ENABLED

        // Each thread will process 'BATCH_SIZE' samples, if the remainder
        // is not 0, 'REMAINDER' threads will process 1 extra sample.
        const uint32_t BATCH_SIZE { sampleCount / THREAD_COUNT };
        const uint32_t REMAINDER { sampleCount % THREAD_COUNT };

        std::vector<std::future<void>> futures;
        futures.reserve(THREAD_COUNT);

        // Process batches in parallel using all available threads.
        for (uint32_t t { 0 }; t < THREAD_COUNT; t++) {
                // Input and output arrays are split for each thread equally, or
                // as close as possible to equally.
                const uint32_t START { t * BATCH_SIZE };
                const uint32_t END { START + BATCH_SIZE + (t < REMAINDER ? 1 : 0) };

                // Emplace back constructs and places the object in free space,
                // which was allocated above
                futures.emplace_back(std::async(std::launch::async, [&]() -> void {
                        for (uint32_t i { START }; i < END; i++) {
                                // All activation values are stored to back-propagate the network.
                                const auto a { new Vector[network.m_layerCount] };
                                a[0] = Vector(network.m_inputSize, inputs);

                                for (uint32_t L { 1 }; L < network.m_layerCount; L++)
                                        a[L] = network.m_L[L - 1]->forward(a[L - 1]);

                                // The cost of the last layer neurons is calculated with (ajL - yj) ^ 2,
                                // this mean that the derivative ∂C is equal to 2 * (ajL - y).
                                Vector dC { (a[network.m_layerCount - 1] - Vector(
                                        network.m_outputSize, outputs + i * network.m_outputSize)) * 2.0f };

                                network.backward(dC, a);
                                delete [] a;
                        }
                }));
        }

        // Wait for all threads to finish
        for (std::future<void> &f : futures)
                f.get();
}

void Train::_PPO(
        // The policy network, also known as the actor.
        Network &policyNetwork,
        // The value network, also known as the critic.
        Network &valueNetwork,
        // A situation dependant environment.
        IEnvironment &environment,
        uint32_t epochs,
        uint32_t maxSteps) {

        struct IterationData {
                uint64_t stateHash;
                float reward;
                float predictedReward;
                Vector policy;
                Vector *policyActivations;
                Vector *valueActivations;
        };

        // Saving variables here for quicker and clearer access
        const uint32_t POLICY_LAYER_COUNT { policyNetwork.m_layerCount };
        const uint32_t VALUE_LAYER_COUNT { valueNetwork.m_layerCount };

        // Keep policies between epochs.
        std::unordered_map<uint64_t, Vector> oldPolicies;

        for (uint32_t i{}; i < epochs; ++i) {
                std::vector<IterationData> iterationData;
                for (uint32_t s{}, done { false }; !done && s < maxSteps; ++s) {
                        iterationData.push_back({});
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData.back();

                        // The current state of the environment, updated
                        // every time IEnvironment::step() is called.
                        const Vector state { environment.getState() };

                        // Save the state hash only since the values won't be used.
                        stateHash = state.hash();

                        // Forward function of the policy network, keeping
                        // activation values.
                        pa = new Vector[POLICY_LAYER_COUNT];
                        pa[0] = state;

                        for (uint32_t L { 1 }; L < POLICY_LAYER_COUNT; L++)
                                pa[L] = policyNetwork.m_L[L - 1]->forward(pa[L - 1]);

                        // Saving the chosen policy in a different vector
                        // as it will substitute the oldPolicies reference.
                        policy = pa[POLICY_LAYER_COUNT - 1];

                        // Forward function of the value network, keeping
                        // activation values. Value are not used right now,
                        // but they will be used later.
                        va = new Vector[VALUE_LAYER_COUNT];
                        va[0] = state;

                        for (uint32_t L { 1 }; L < VALUE_LAYER_COUNT; L++)
                                va[L] = valueNetwork.m_L[L - 1]->forward(va[L - 1]);

                        // Saving the predicted rewards in a different vector
                        // to make the next steps easier.
                        predictedReward = va[VALUE_LAYER_COUNT - 1].at(0);

                        // Step the Environment forward, updating the state.
                        auto [_reward, _done] { environment.step(pa[POLICY_LAYER_COUNT - 1]) };
                        reward = _reward;
                        done = _done;
                }

#ifdef DEBUG_MODE_ENABLED
                Log << LOGGER_PREF(DEBUG) << "Execution [" << i << "] done.\n";
#endif // DEBUG_MODE_ENABLED

                const uint32_t STATE_COUNT { (uint32_t)iterationData.size() };
                float *advantages { new float[STATE_COUNT] }, previous { 0.0f };

                // Compute advantages before the training as the calculation is done
                // backwards to use the previous advantage as a part of the formula.
                for (int32_t s { (int32_t)STATE_COUNT - 1 }; s >= 0; s--) {
                        const float reward { iterationData[s].reward };
                        const float predictedReward { iterationData[s].predictedReward };
                        const float nextPredictedReward { s == STATE_COUNT - 1 ? 0 : iterationData[s + 1].predictedReward };

                        const float delta { reward + GAMMA * nextPredictedReward * predictedReward };
                        advantages[s] = delta + GAMMA * LAMBDA * previous;
                        previous = advantages[s];
                }

                // Trains the networks
                for (uint32_t s { 0 }; s < STATE_COUNT; s++) {
                        auto &[stateHash, reward, predictedReward, policy, pa, va] = iterationData[s];

                        // Save content of arrays in variables for quicker access.
                        const float advantage { advantages[s] };
                        policy = policy.max(EPSILON);

                        // Value network cost, the cost in a PPO environment is a single
                        // value back-propagated throughout all the output neurons.
                        Vector vdC { 2.0f * (predictedReward - reward) };

                        // Train the value network and delete dynamically allocated array.
                        valueNetwork.backward(vdC, va);
                        delete [] va;

                        // If the current state has never been met, the old policy
                        // is set to the current policy.
                        oldPolicies.insert({stateHash, policy});

                        //               ⌈      π𝜃ɴᴇᴡ(aₜ|sₜ)                           ⌉
                        // Lᴘᴏʟɪᴄʏ(𝜃) = 𝔼| min( ⎯⎯⎯⎯⎯⎯⎯⎯ Aₜ, clip(p, 1 - ε, 1 + ε)Aₜ) |
                        //               ⌊      π𝜃ᴏʟᴅ(aₜ|sₜ)                           ⌋
                        const Vector ratio { policy / oldPolicies.at(stateHash) };
                        const Vector clippedRatio { ratio.clamp(1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON) };
                        const Vector surrogateLoss { (ratio * advantage).min(clippedRatio * advantage) };

                        // Replace the old policy with the new one, it is important to
                        // save the policy after .max(EPSILON) is done to prevent division
                        // by 0 in the 'ratio' calculation.
                        oldPolicies[stateHash] = policy;

                        // Train the policy network and delete dynamically allocated array.
                        policyNetwork.backward(surrogateLoss, pa);
                        delete [] pa;
                }

                // Delete advantages and reset environment in preparation for
                // next epoch.
                delete [] advantages;
                environment.reset();

#ifdef DEBUG_MODE_ENABLED
                Log << LOGGER_PREF(DEBUG) << "Training [" << i << "] done.\n";
#endif // DEBUG_MODE_ENABLED
        }
}