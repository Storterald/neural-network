#include "Train.h"

#include <ranges>

#include "Base.h"

void Train::supervisedTraining(
        Network &network,
        uint32_t sampleCount,
        const float *inputs,
        const float *outputs
) {
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
                                Vector dC { (a[network.m_layerCount - 1] - Vector(network.m_outputSize, outputs + i * network.m_outputSize)) * 2.0f };

                                network.backward(dC, a);
                                delete [] a;
                        }
                }));
        }

        // Wait for all threads to finish
        for (std::future<void> &f : futures)
                f.get();
}

Network Train::_createStateApproximatorNetwork(
        uint32_t stateSize,
        uint32_t actionSize,
        IEnvironment &environment
) {
        constexpr uint32_t APPROXIMATOR_NETWORK_SIZE = 3;
        LayerCreateInfo approximatorInfos[APPROXIMATOR_NETWORK_SIZE - 1] {
                {FULLY_CONNECTED, TANH, 32},
                // The approximated state size must have a reasonable size for every possible input
                // size, one way to generalize it is using a logarithmic formula.
                //             S
                // Sᵣ = ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
                //      max(1, log₁₀(S))
                {FULLY_CONNECTED, TANH, stateSize / (uint32_t)std::max(1.0f, (float)std::log10(stateSize)) }
        };
        // Scales down the size of a state, decreasing the required memory
        // required to store the states
        Network stateApproximator(stateSize, 2, approximatorInfos, BASE_PATH "/approximator.nnv");

        constexpr uint32_t DECODER_NETWORK_SIZE = 3;
        LayerCreateInfo decoderInfos[DECODER_NETWORK_SIZE - 1] {
                {FULLY_CONNECTED, TANH, 32},
                {FULLY_CONNECTED, TANH, stateSize }
        };
        // Helper network to train the state approximator.
        Network stateDecoder(stateApproximator.m_outputSize, 2, decoderInfos, "");

        for (uint32_t i { 0 }; i < APPROXIMATOR_TRAINING_EPOCHS; i++) {
                for (uint32_t j { 0 }, done { false }; !done && j < APPROXIMATOR_TRAINING_MAX_STEPS; ++j) {
                        const Vector state { environment.getState() };

                        // Forward function of the state approximator network, keeping
                        // activation values.
                        const auto va { new Vector[APPROXIMATOR_NETWORK_SIZE] };
                        va[0] = state;

                        for (uint32_t L { 1 }; L < APPROXIMATOR_NETWORK_SIZE; L++)
                                va[L] = stateApproximator.m_L[L - 1]->forward(va[L - 1]);

                        // Forward function of the state decoder network, keeping
                        // activation values.
                        const auto vd { new Vector[DECODER_NETWORK_SIZE] };
                        vd[0] = va[APPROXIMATOR_NETWORK_SIZE - 1];

                        for (uint32_t L { 1 }; L < DECODER_NETWORK_SIZE; L++)
                                vd[L] = stateDecoder.m_L[L - 1]->forward(vd[L - 1]);

                        // Backward step on the encoder,
                        const Vector approximatorCost { (va[DECODER_NETWORK_SIZE - 1] - vd[DECODER_NETWORK_SIZE - 1]) * 2.0f };
                        stateApproximator.backward(approximatorCost, va);

                        // Backward step on the decoder, using the decoded state
                        // and the original one.
                        const Vector decoderCost { (vd[DECODER_NETWORK_SIZE - 1] - state) * 2.0f };
                        stateDecoder.backward(decoderCost, vd);

                        delete [] va;
                        delete [] vd;

                        Vector action(actionSize);
                        // Selecting a random action
                        action[rand() % actionSize] = 1.0f;

                        auto [_, _done] { environment.step(action) };
                        done = (uint32_t)_done;
                }
                environment.reset();
        }

#ifdef DEBUG_MODE_ENABLED
        Log << LOGGER_PREF(DEBUG) << "Approximation done.\n";
#endif // DEBUG_MODE_ENABLED

        return stateApproximator;
}

static inline uint64_t vectorHash(const Vector& vec)
{
        uint64_t hash = 0;
        for (uint32_t i { 0 }; i < vec.size(); i++)
                hash ^= std::hash<float>{}(vec.at(i)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

        return hash;
}

void Train::_PPOTraining(
        // The policy network, also known as the actor.
        Network &policyNetwork,
        // The value network, also known as the critic.
        Network &valueNetwork,
        // A situation dependant environment.
        IEnvironment &environment,
        std::unordered_map<uint64_t , Vector> &oldPolicies,
        const Network *pStateApproximator
) {
        // Saving variables here for quicker access
        const uint32_t POLICY_LAYER_COUNT { policyNetwork.m_layerCount };
        const uint32_t VALUE_LAYER_COUNT { valueNetwork.m_layerCount };

        // The stored states used for mapping.
        std::vector<Vector> states;
        std::vector<float> rewards;                     // [states.size()]
        std::vector<float> valueRewards;                // [states.size()]
        std::vector<Vector> policies;                   // [states.size()]

        // Needed when back-propagating the networks.
        std::vector<Vector *> policyActivations;        // [states.size()]
        std::vector<Vector *> valueActivations;         // [states.size()]

        // Runs the Environment from start to finish.
        for (uint32_t j { 0 }, done { false }; !done && j < PPO_TRAINING_MAX_STEPS; ++j) {
                // The current state of the environment, updated
                // every time IEnvironment::step() is called.
                const Vector state { environment.getState() };

                // Saving an approximated smaller state if approximator network is given
                if (pStateApproximator)
                        states.push_back(pStateApproximator->forward(state));
                else
                        states.push_back(state);

                // Forward function of the policy network, keeping
                // activation values.
                policyActivations.push_back(new Vector[POLICY_LAYER_COUNT]);
                Vector *const pa { policyActivations.back() };
                pa[0] = state;

                for (uint32_t L { 1 }; L < POLICY_LAYER_COUNT; L++)
                        pa[L] = policyNetwork.m_L[L - 1]->forward(pa[L - 1]);

                // Saving the chosen policy in a different vector
                // as it will substitute the oldPolicies reference.
                policies.push_back(pa[POLICY_LAYER_COUNT - 1]);

                // Forward function of the value network, keeping
                // activation values. Value are not used right now,
                // but they will be used later.
                valueActivations.push_back(new Vector[VALUE_LAYER_COUNT]);
                Vector *const va { valueActivations.back() };
                va[0] = state;

                for (uint32_t L { 1 }; L < VALUE_LAYER_COUNT; L++)
                        va[L] = valueNetwork.m_L[L - 1]->forward(va[L - 1]);

                // Saving the predicted rewards in a different vector
                // to make the next steps easier.
                valueRewards.push_back(va[VALUE_LAYER_COUNT - 1].at(0));

                // Step the Environment forward, updating the state.
                auto [reward, _done] { environment.step(pa[POLICY_LAYER_COUNT - 1]) };
                rewards.push_back(reward);
                done = _done;
        }
#ifdef DEBUG_MODE_ENABLED
        Log << LOGGER_PREF(DEBUG) << "Execution done.\n";
#endif // DEBUG_MODE_ENABLED

        const uint32_t STATE_COUNT { (uint32_t)states.size() };

        // Compute advantages
        float *advantages { new float[STATE_COUNT] };
        float previous { 0.0f };
        for (int32_t s { (int32_t)STATE_COUNT - 1 }; s >= 0; s--) {
                const float delta { rewards[s] + GAMMA * (s == STATE_COUNT - 1 ? 0 : valueRewards[s + 1]) - valueRewards[s] };
                advantages[s] = delta + GAMMA * LAMBDA * previous;
                previous = advantages[s];
#ifdef DEBUG_MODE_ENABLED
                Log << LOGGER_PREF(DEBUG) << "Advantage [" << s << "]: " << advantages[s] << "\n";
#endif // DEBUG_MODE_ENABLED
        }

        // Trains the networks
        for (uint32_t s { 0 }; s < STATE_COUNT; s++) {
                const uint64_t stateHash { vectorHash(states[s]) };
                const Vector policy { policies[s].max(EPSILON) };

                const float advantage { advantages[s] };

                // Value network cost
                Vector vdC(1);
                vdC[0] = 2.0f * (valueRewards[s] - rewards[s]);

                // Train the value network.
                valueNetwork.backward(vdC, valueActivations[s]);

                // If the current state has never been met, the old policy
                // is set to the current policy.
                oldPolicies.insert({stateHash, policy});

#ifdef DEBUG_MODE_ENABLED
                Log << LOGGER_PREF(DEBUG) << "[" << s << "]: hash: " << stateHash << ", new policy: " << policy << ", old policy: " << oldPolicies[stateHash] << "\n";
#endif // DEBUG_MODE_ENABLED

                //               ⌈      π𝜃ɴᴇᴡ(aₜ|sₜ)                           ⌉
                // Lᴘᴏʟɪᴄʏ(𝜃) = 𝔼| min( ⎯⎯⎯⎯⎯⎯⎯⎯ Aₜ, clip(p, 1 - ε, 1 + ε)Aₜ) |
                //               ⌊      π𝜃ᴏʟᴅ(aₜ|sₜ)                           ⌋
                const Vector ratio { policy / oldPolicies.at(stateHash) };
                const Vector clippedRatio { ratio.clamp(1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON) };
                const Vector surrogateLoss { (ratio * advantage).min(clippedRatio * advantage) };

                // Replace the old policy with the new one
                oldPolicies[stateHash] = policy;

                policyNetwork.backward(surrogateLoss, policyActivations[s]);

#ifdef DEBUG_MODE_ENABLED
                Log << LOGGER_PREF(DEBUG) << "Iteration [" << s << "]. Ratio: " << ratio << " clipped: " << clippedRatio << " loss: " << surrogateLoss << "\n";
#endif // DEBUG_MODE_ENABLED
        }

        // Delete allocated data
        delete [] advantages;
        for (const Vector *a : policyActivations)
                delete [] a;
        for (const Vector *a : valueActivations)
                delete [] a;

        // Reset environment
        environment.reset();
}