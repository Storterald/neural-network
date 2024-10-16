#pragma once

extern "C" {
#include "SIMD.h"
}

namespace Intrinsic {

        [[nodiscard]] inline SIMD support()
        {
                const static SIMD support { getSIMDSupport() };
                return support;
        }

}