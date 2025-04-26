#pragma once

enum SIMD {
        SIMD_UNSUPPORTED,
        SIMD_SSE3,
        SIMD_AVX,
        SIMD_AVX512
};

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
