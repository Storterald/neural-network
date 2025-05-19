#pragma once

#include <cstdint>

#include <neural-network/base.h>

namespace nn {

enum _math_type : uint32_t {
        MATH_NORMAL =  0
#ifdef IS_X86_64BIT
        , MATH_SSE3 =  4,
        MATH_AVX    =  8,
        MATH_AVX512 = 16
#endif // IS_X86_64BIT
#ifdef BUILD_CUDA_SUPPORT
        , MATH_CUDA = (uint32_t)-1
#endif // BUILD_CUDA_SUPPORT
}; // enum math_type

} // namespace nn