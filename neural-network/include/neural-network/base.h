#pragma once

#if !defined(NDEBUG) && !defined(DEBUG_MODE_ENABLED)
#define DEBUG_MODE_ENABLED
#endif // !NDEBUG && !DEBUG_MODE_ENABLED

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

namespace nn {

#ifdef BUILD_CUDA_SUPPORT
using stream = cudaStream_t;
#else // BUILD_CUDA_SUPPORT
using stream = void *;
#endif // BUILD_CUDA_SUPPORT
constexpr stream invalid_stream = stream{};

constexpr float EPSILON       = 1e-8f;
constexpr float LEARNING_RATE = 5e-5f;
constexpr float CLIP_EPSILON  = 0.2f;
constexpr float GAMMA         = 0.99f;
constexpr float LAMBDA        = 0.95f;

} // namespace nn
