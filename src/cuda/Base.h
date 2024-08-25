#pragma once

#include "../Base.h"

#ifndef USE_CUDA
#error "To include cuda/Base.h, it is required to compile with CUDA support."
#endif

#ifdef DEBUG_MODE_ENABLED
#include "../utils/Logger.h"
#endif // DEBUG_MODE_ENABLED

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// No checks in release mode to increase speed.
#ifdef DEBUG_MODE_ENABLED
__host__ inline void checkCudaError(
        cudaError_t result,
        const char *msg
) {
        if (result != cudaSuccess)
                throw Logger::fatal_error(std::string(msg) + " CUDA code: " + cudaGetErrorString(result));
}
#define CUDA_CHECK_ERROR(...) checkCudaError(__VA_ARGS__)
#else
#define CUDA_CHECK_ERROR(f, ...) f
#endif // DEBUG_MODE_ENABLED