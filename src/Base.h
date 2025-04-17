#pragma once

#include <cstdint>
#include <string_view>

#if !defined(NDEBUG) && !defined(DEBUG_MODE_ENABLED)
#define DEBUG_MODE_ENABLED
#endif // !NDEBUG && !DEBUG_MODE_ENABLED

// Generic constants.
constexpr float EPSILON { 1e-8f };
constexpr float LEARNING_RATE { 5e-5f };
constexpr uint32_t MAX_FILE_SIZE { (uint32_t)1e9f };