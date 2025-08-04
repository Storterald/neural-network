#pragma once

#if !defined(NDEBUG) && !defined(DEBUG_MODE_ENABLED)
#define DEBUG_MODE_ENABLED
#endif // !NDEBUG && !DEBUG_MODE_ENABLED

#include <type_traits>

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

namespace nn {

template<typename T, typename ...Ts>
concept is_any = std::disjunction_v<std::is_same<T, Ts>...>;

template<typename T>
concept floating_type = is_any<T, float>;  // TODO float128, float64, float16, int4, bit

#ifdef BUILD_CUDA_SUPPORT
using stream_t = cudaStream_t;
#else // BUILD_CUDA_SUPPORT
using stream_t = void *;
#endif // BUILD_CUDA_SUPPORT
constexpr stream_t invalid_stream = stream_t{};

constexpr float EPSILON       = 1e-8f;
constexpr float LEARNING_RATE = 5e-5f;
constexpr float CLIP_EPSILON  = 0.2f;
constexpr float GAMMA         = 0.99f;
constexpr float LAMBDA        = 0.95f;

} // namespace nn
