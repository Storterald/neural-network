#pragma once

#include <neural-network/base.h>

namespace nn {

enum simd {
        SIMD_UNSUPPORTED,
        SIMD_SSE3,
        SIMD_AVX,
        SIMD_AVX512

}; // enum simd

namespace intrinsic {

#if IS_X86_64BIT
        extern "C" simd _get_simd_support();
#endif // IS_X86_64BIT

        [[nodiscard]] inline simd support()
        {
#if IS_X86_64BIT
                const static simd support = _get_simd_support();
                return support;
#else // IS_X86_64BIT
                return SIMD_UNSUPPORTED;
#endif // IS_X86_64BIT
        }

} // namespace intrinsic

} // namespace nn
