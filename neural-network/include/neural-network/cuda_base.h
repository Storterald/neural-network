#pragma once

#include <cstdint>

#include <neural-network/base.h>

#ifdef DEBUG_MODE_ENABLED
#include <cuda_runtime.h>

#include <string>

#include <neural-network/utils/logger.h>
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

namespace nn {

constexpr uint32_t CUDA_THREADS = 256;
constexpr uint32_t CUDA_MINIMUM = 10000;

} // namespace nn
