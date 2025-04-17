#pragma once

enum SIMD {
        SIMD_UNSUPPORTED,
        SIMD_SSE,
        SIMD_AVX,
        SIMD_AVX512
};

extern "C" {
        // Include assembly function
        extern SIMD getSIMDSupport();
}

namespace Intrinsic {

        [[nodiscard]] inline SIMD support()
        {
                const static SIMD support { getSIMDSupport() };
                return support;
        }

}