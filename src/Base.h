#pragma once

#include <cstdint>
#include <string_view>

#include "utils/Logger.h"

#if !defined(NDEBUG) && !defined(DEBUG_MODE_ENABLED)
#define DEBUG_MODE_ENABLED
#endif // !NDEBUG && !DEBUG_MODE_ENABLED

// Generic constants.
constexpr float EPSILON       = 1e-8f;
constexpr float LEARNING_RATE = 5e-5f;