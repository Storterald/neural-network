#pragma once

#include <cstdint>

#include "utils/Logger.h"
#include "utils/Macros.h"

#if !defined(NDEBUG) && !defined(DEBUG_MODE_ENABLED)
#define DEBUG_MODE_ENABLED
#endif // !NDEBUG && !DEBUG_MODE_ENABLED

#ifdef BUILD_CUDA_SUPPORT
#include "CudaBase.h"
#endif // BUILD_CUDA_SUPPORT

// Generic constants.
constexpr float EPSILON       = 1e-8f;
constexpr float LEARNING_RATE = 5e-5f;