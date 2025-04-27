#pragma once

namespace nn {

enum SIMD {
        SIMD_UNSUPPORTED,
        SIMD_SSE3,
        SIMD_AVX,
        SIMD_AVX512

}; // enum SIMD

extern "C" {
        extern SIMD get_SIMD_support();
}

namespace Intrinsic {

        [[nodiscard]] inline SIMD support()
        {
                const static SIMD support = get_SIMD_support();
                return support;
        }

} // namespace Intrinsic

} // namespace nn