#pragma once

namespace nn {

enum simd {
        SIMD_UNSUPPORTED,
        SIMD_SSE3,
        SIMD_AVX,
        SIMD_AVX512

}; // enum simd

extern "C" simd _get_simd_support();

namespace intrinsic {

        [[nodiscard]] inline simd support()
        {
                const static simd support = _get_simd_support();
                return support;
        }

} // namespace intrinsic

} // namespace nn
