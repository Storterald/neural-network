#pragma once

#include <neural-network/base.h>

namespace nn {

enum simd_support {
        SIMD_UNSUPPORTED,
        SIMD_SSE3,
        SIMD_AVX,
        SIMD_AVX512

}; // enum simd

namespace intrinsic {

#if TARGET_X86_64
        extern "C" simd_support _get_simd_support();
#endif // TARGET_X86_64

        [[nodiscard]] inline simd_support support()
        {
#if TARGET_X86_64
                const static simd_support support = _get_simd_support();
                return support;
#else // TARGET_X86_64
                return SIMD_UNSUPPORTED;
#endif // TARGET_X86_64
        }

} // namespace intrinsic

} // namespace nn
