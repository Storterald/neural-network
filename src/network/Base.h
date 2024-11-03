#pragma once

#include <cstdint>

#include "../math/Base.h"
#include "../math/Math.h"
#include "../math/CudaMath.h"

#include "../utils/Logger.h"

// PPO related constants.
constexpr float CLIP_EPSILON { 2e-1f };
constexpr float GAMMA { 0.99f };
constexpr float LAMBDA { 0.95f };
constexpr uint32_t APPROXIMATOR_TRAINING_MAX_STEPS { 1000 };
constexpr uint32_t PPO_TRAINING_MAX_STEPS { 1000 };
constexpr uint32_t APPROXIMATOR_TRAINING_EPOCHS { 100 };