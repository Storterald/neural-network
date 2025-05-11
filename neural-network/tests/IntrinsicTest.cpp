#include <gtest/gtest.h>

#include <neural-network/intrinsic/intrinsic.h>
#include <neural-network/base.h>

#ifdef IS_X86_64BIT
#include <intrin.h>
#endif // IS_X86_64BIT

TEST(IntrinsicTest, CpuSIMDSupportIsCorrectlyDetected) {
#ifdef IS_X86_64BIT
        constexpr int EAX = 0;
        constexpr int EBX = 1;
        constexpr int ECX = 2;
        constexpr int EDX = 3;

        const nn::simd support = []() -> nn::simd {
                int regs[4]{};
                __cpuid(regs, 7);
                if (regs[EBX] & (1 << 16))
                        return nn::SIMD_AVX512;

                __cpuid(regs, 1);

                if (regs[ECX] & (1 << 12))
                        return nn::SIMD_AVX;

                if (regs[ECX] & (1 << 0))
                        return nn::SIMD_SSE3;

                return nn::SIMD_UNSUPPORTED;
        }();
#else // IS_X86_64BIT
        constexpr nn::simd support = nn::SIMD_UNSUPPORTED;
#endif // IS_X86_64BIT
        EXPECT_EQ(nn::intrinsic::_get_simd_support(), support);
}
