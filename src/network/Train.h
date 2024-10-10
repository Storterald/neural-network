#pragma once

#include <future>

#include "Network.h"
#include "PPO/IEnvironment.h"

// Train is a class and not a namespace because it needs
// to be a friend of the Network class.
class Train {
public:
        static void supervisedTraining(
                Network &network,
                uint32_t sampleCount,
                const float *inputs,
                const float *outputs
        );

        template<std::derived_from<IEnvironment> Environment>
        static void PPOTraining(
                Network &policyNetwork,
                Network &valueNetwork,
                uint32_t epochs
        ) {
                Environment environment{};
                // Keep policies between iterations
                std::unordered_map<uint64_t, Vector> oldPolicies;

                // The smallest input size the approximator reduces is 100, this avoids
                // useless calls to a redundant network.
                if (policyNetwork.m_inputSize >= 100) {
                        Network stateApproximator { _createStateApproximatorNetwork(policyNetwork.m_inputSize, policyNetwork.m_outputSize, environment)};
                        for (uint32_t i { 0 }; i < epochs; i++)
                                _PPOTraining(policyNetwork, valueNetwork, environment, oldPolicies, &stateApproximator);
                } else {
                        // For small states run without stateApproximator
                        for (uint32_t i { 0 }; i < epochs; i++)
                                _PPOTraining(policyNetwork, valueNetwork, environment, oldPolicies, nullptr);
                }
        }

private:
        static Network _createStateApproximatorNetwork(
                uint32_t stateSize,
                uint32_t actionSize,
                IEnvironment &environment
        );

        static void _PPOTraining(
                Network &policyNetwork,
                Network &valueNetwork,
                IEnvironment &environment,
                std::unordered_map<uint64_t , Vector> &oldPolicies,
                const Network *pStateApproximator
        );

}; // class Train