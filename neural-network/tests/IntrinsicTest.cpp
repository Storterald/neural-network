#include <gtest/gtest.h>

#include <neural-network/intrinsic/intrinsic.h>
#include <neural-network/base.h>

#ifdef IS_X86_64BIT
#ifdef _MSC_VER
#include <intrin.h>
#else // _MSC_VER
#include <cpuid.h>
#endif // _MSC_VER

static nn::simd_support _compiler_cpuid_simd_support() {
#ifdef _MSC_VER
        constexpr int EBX = 1;
        constexpr int ECX = 2;

        int regs[4]{};
        __cpuid(regs, 7);
        if (regs[EBX] & 0b1000000000000000)
                return nn::SIMD_AVX512;

        __cpuid(regs, 1);

        if (regs[ECX] & 0b100000000000)
                return nn::SIMD_AVX;

        if (regs[ECX] & 0b1)
                return nn::SIMD_SSE3;

        return nn::SIMD_UNSUPPORTED;
#else
        uint32_t eax, ebx, ecx, edx;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        if (ebx & bit_AVX512DQ)
                return nn::SIMD_AVX512;

        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        if (ecx & bit_FMA)
                return nn::SIMD_AVX;

        if (ecx & bit_SSE3)
                return nn::SIMD_SSE3;

        return nn::SIMD_UNSUPPORTED;
#endif // _MSC_VER
}
#endif // IS_X86_64BIT

TEST(IntrinsicTest, CpuSIMDSupportIsCorrectlyDetected) {
#ifdef IS_X86_64BIT
        const nn::simd_support support = _compiler_cpuid_simd_support();
#else // IS_X86_64BIT
        constexpr nn::simd_support support = nn::SIMD_UNSUPPORTED;
#endif // IS_X86_64BIT
        EXPECT_EQ(nn::intrinsic::_get_simd_support(), support);
}
