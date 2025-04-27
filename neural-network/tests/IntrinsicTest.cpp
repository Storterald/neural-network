#include <gtest/gtest.h>

#include <intrin.h>

#include <neural-network/intrinsic/Intrinsic.h>
#include <neural-network/Base.h>

USE_NN

TEST(IntrinsicTest, CpuSIMDSupportIsCorrectlyDetected) {
        constexpr int EAX = 0;
        constexpr int EBX = 1;
        constexpr int ECX = 2;
        constexpr int EDX = 3;

        const SIMD support = []() -> SIMD {
                int regs[4]{};
                __cpuid(regs, 7);
                if (regs[EBX] & (1 << 16))
                        return SIMD_AVX512;

                __cpuid(regs, 0);

                if (regs[ECX] & (1 << 12))
                        return SIMD_AVX;

                if (regs[ECX] & (1 << 0))
                        return SIMD_SSE3;

                return SIMD_UNSUPPORTED;
        }();

        EXPECT_EQ(get_SIMD_support(), support);
}
