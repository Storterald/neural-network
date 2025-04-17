#include <gtest/gtest.h>

#include <intrin.h>

#include "../src/intrinsic/Intrinsic.h"

TEST(SIMDSupport, GetSIMDSupportFunctionReturnsCorrectValue) {
        SIMD support { SIMD_SSE };

        int cpuInfo[4]{};
        __cpuid(cpuInfo, 1);

        if ((cpuInfo[3] & (1 << 25)) == 0 || (cpuInfo[3] & (1 << 26)) == 0)
                support = SIMD_UNSUPPORTED;
        else if ((cpuInfo[2] & (1 << 27)) != 0 && (cpuInfo[2] & (1 << 28)) != 0) {
                unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
                if ((xcrFeatureMask & 0x6) == 0x6) {
                        __cpuid(cpuInfo, 7);
                        if ((xcrFeatureMask & 0xE6) == 0xE6 && (cpuInfo[1] & (1 << 16)) != 0)
                                support = SIMD_AVX512;
                        else
                                support = SIMD_AVX;
                }
        }

        EXPECT_EQ(Intrinsic::support(), support);
}