#pragma once

#include <cstdint>

#include <neural-network/Base.h>

#ifdef DEBUG_MODE_ENABLED
#include <cuda_runtime.h>

#include <string>

#include <neural-network/utils/Logger.h>
#endif // DEBUG_MODE_ENABLED

#ifndef BUILD_CUDA_SUPPORT
#error CudaBase.h cannot be included without BUILD_CUDA_SUPPORT
#endif // !BUILD_CUDA_SUPPORT

// No checks in release mode to increase speed.
#ifdef DEBUG_MODE_ENABLED
#define CUDA_CHECK_ERROR(__result__, __msg__)                                               \
        do {                                                                                \
                if (auto code = __result__; code != cudaSuccess)                            \
                        throw LOGGER_EX(std::string(__msg__)                                \
                                + " \"" + cudaGetErrorString(code) + "\"");                 \
        }                                                                                   \
        while(false)
#else // DEBUG_MODE_ENABLED
#define CUDA_CHECK_ERROR(f, ...) f
#endif // DEBUG_MODE_ENABLED

NN_BEGIN

// Math related constants.
constexpr uint32_t BLOCK_BITSHIFT = 8;
constexpr uint32_t BLOCK_SIZE = 1 << BLOCK_BITSHIFT;

// Since copying data to the GPU and back, and launching CUDA kernels
// has a cost, for small amount of data it's not worth it to use CUDA.
constexpr uint32_t CUDA_MINIMUM = 10000;

NN_END
