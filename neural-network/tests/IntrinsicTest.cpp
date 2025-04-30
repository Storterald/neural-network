#include <gtest/gtest.h>

#include <intrin.h>

#include <neural-network/intrinsic/intrinsic.h>

TEST(IntrinsicTest, CpuSIMDSupportIsCorrectlyDetected) {
        constexpr int EAX = 0;
        constexpr int EBX = 1;
        constexpr int ECX = 2;
        constexpr int EDX = 3;

        const nn::simd support = []() -> nn::simd {
                int regs[4]{};
                __cpuid(regs, 7);
                if (regs[EBX] & (1 << 16))
                        return nn::SIMD_AVX512;

                __cpuid(regs, 0);

                if (regs[ECX] & (1 << 12))
                        return nn::SIMD_AVX;

                if (regs[ECX] & (1 << 0))
                        return nn::SIMD_SSE3;

                return nn::SIMD_UNSUPPORTED;
        }();

        EXPECT_EQ(nn::_get_simd_support(), support);
}
