#pragma once

#include <future>

#include "Network.h"

struct Train {
        static void supervisedTraining(
                Network &network,
                uint32_t sampleCount,
                const float *inputs,
                const float *outputs
        );

        static void PPOTraining(
                Network &policyNetwork,
                Network &valueNetwork,
                uint32_t sampleCount,
                const float *inputs
        );

};