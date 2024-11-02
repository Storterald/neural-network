#pragma once

enum SIMD {
        UNSUPPORTED,
        SSE,
        AVX,
        AVX512
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