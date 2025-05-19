#pragma once

#include <cstdint>
#include <type_traits>

#include "_math_normal.h"

#ifdef BUILD_CUDA_SUPPORT
#include "_math_cuda.h"
#endif // BUILD_CUDA_SUPPORT

namespace nn {

#ifdef BUILD_CUDA_SUPPORT
template<_math_type T>
using _math = std::conditional_t<T == MATH_CUDA, _math_cuda, _math_normal<T>>;
#else
template<_math_type T>
using _math = _math_normal<T>;
#endif // BUILD_CUDA_SUPPORT

} // namespace nn