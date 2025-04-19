#pragma once

#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../Base.h"

// No checks in release mode to increase speed.
#ifdef DEBUG_MODE_ENABLED
#define CUDA_CHECK_ERROR(RESULT, MESSAGE)                                                   \
        do {                                                                                \
                if (auto code = RESULT; code != cudaSuccess)                                \
                        throw Logger::fatal_error(LOGGER_PREF(FATAL) + std::string(MESSAGE) \
                                + " \"" + cudaGetErrorString(code) + "\"");                 \
        }                                                                                   \
        while(false)
#else
#define CUDA_CHECK_ERROR(f, ...) f
#endif // DEBUG_MODE_ENABLED

// Math related constants.
constexpr uint32_t BLOCK_BITSHIFT { 8 };
constexpr uint32_t BLOCK_SIZE { 1 << BLOCK_BITSHIFT };

// Since copying data to the GPU and back, and launching CUDA kernels
// has a cost, for small amount of data it's not worth it to use CUDA.
constexpr uint32_t CUDA_MINIMUM { 10000 };
