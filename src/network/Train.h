#pragma once

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

        template<std::derived_from<IEnvironment> Environment, typename ...Ts>
        static void PPOTraining(
                Network &policyNetwork,
                Network &valueNetwork,
                uint32_t epochs,
                uint32_t maxSteps,
                Ts ...args
        ) {
                Environment env { args... };
                _PPOTraining(policyNetwork, valueNetwork, env, epochs, maxSteps);
        }

private:
        static void _PPOTraining(
                Network &policyNetwork,
                Network &valueNetwork,
                IEnvironment &environment,
                uint32_t epochs,
                uint32_t maxSteps
        );

}; // class Train