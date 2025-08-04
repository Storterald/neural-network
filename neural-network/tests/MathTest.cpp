#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <cmath>

#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>

#if SIMD_SUPPORT_LEVEL   >= 1
#include <neural-network/utils/_simd128.h>
#endif // SIMD_SUPPORT_LEVEL >= 1

#if SIMD_SUPPORT_LEVEL >= 2
#include <neural-network/utils/_simd256.h>
#endif // SIMD_SUPPORT_LEVEL >= 2

#if SIMD_SUPPORT_LEVEL >= 3
#include <neural-network/utils/_simd512.h>
#endif // SIMD_SUPPORT_LEVEL >= 3

#include "../src/math/_math_normal.h"

#ifdef TARGET_X86_64
#include "../src/math/_math_simd.h"
namespace simd = nn::simd;
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
#include "../src/math/_math_cuda.h"
#endif // BUILD_CUDA_SUPPORT

#include "printers.h"

#define EXPECT_EQ_FLOAT_VEC(expected, actual, thresh)                   \
do {                                                                    \
        for (uint32_t idx = 0; idx < (expected).size(); ++idx)          \
                EXPECT_NEAR((expected)[idx], (actual)[idx], thresh);    \
} while (false)

TEST(MathTest, Sum) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_normal::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SumLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SumMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, SumPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SumLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SumMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, SumPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SumLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SumMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Sum) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::sum(
                COUNT, v1.data(nn::loc_type::device),
                v2.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), v1.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Sub) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_normal::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SubLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SubMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, SubPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SubLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SubMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, SubPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SubLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SubMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Sub) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::sub(
                COUNT, v1.data(nn::loc_type::device),
                v2.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), v1.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Mul) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_normal::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MulLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MulMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MulPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MulLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MulMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MulPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MulLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MulMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Mul) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::mul(
                COUNT, v1.data(nn::loc_type::device),
                v2.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), v1.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Div) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_normal::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, DivLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, DivMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, DivPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, DivLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, DivMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, DivPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, DivLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, DivMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Div) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::div(
                COUNT, v1.data(nn::loc_type::device),
                v2.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), v1.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, SumScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SumScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SumScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, SumScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SumScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SumScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, SumScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SumScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SumScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, SumScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar       = 3.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::sum(
                COUNT, data.data(nn::loc_type::device),
                scalar, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, SubScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SubScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, SubScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, SubScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SubScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, SubScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, SubScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SubScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, SubScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, SubScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar       = 3.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::sub(
                COUNT, data.data(nn::loc_type::device),
                scalar, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, MulScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MulScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MulScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MulScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MulScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MulScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MulScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MulScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MulScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, MulScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar       = 3.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::mul(
                COUNT, data.data(nn::loc_type::device),
                scalar, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, DivScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, DivScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, DivScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, DivScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, DivScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, DivScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, DivScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, DivScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, DivScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::_m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, DivScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar       = 3.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::div(
                COUNT, data.data(nn::loc_type::device),
                scalar, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Tanh) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_normal::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, TanhLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, TanhMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, TanhPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, TanhLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, TanhMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, TanhPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, TanhLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, TanhMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Tanh) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::tanh(
                COUNT, data.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, TanhDerivative) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_normal::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativePrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, TanhDerivativeLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, TanhDerivativeMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, TanhDerivativePrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, TanhDerivativeLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, TanhDerivativeMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, TanhDerivativePrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, TanhDerivativeLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, TanhDerivativeMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, TanhDerivative) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::tanh_derivative(
                COUNT, data.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, ReLU) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_normal::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ReLULess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ReLUMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, ReLUPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ReLULess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ReLUMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, ReLUPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ReLULess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ReLUMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, ReLU) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::ReLU(
                COUNT, data.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], 0.0f);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, ReLUDerivative) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_normal::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativePrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ReLUDerivativeLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ReLUDerivativeMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, ReLUDerivativePrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ReLUDerivativeLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ReLUDerivativeMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, ReLUDerivativePrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ReLUDerivativeLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ReLUDerivativeMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::_m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, ReLUDerivative) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::ReLU_derivative(
                COUNT, data.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, MinScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m128>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MinScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m128>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MinScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m128>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MinScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m256>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MinScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m256>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MinScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m256>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MinScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m512>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MinScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m512>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MinScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m512>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, MinScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min          = 3.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::min(
                COUNT, data.data(nn::loc_type::device),
                min, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min((float)data[i], min);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, MaxScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m128>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MaxScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m128>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MaxScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m128>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MaxScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m256>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MaxScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m256>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MaxScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m256>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MaxScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m512>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MaxScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m512>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MaxScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m512>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, MaxScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float max          = 3.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::max(
                COUNT, data.data(nn::loc_type::device),
                max, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], max);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, ClampScalar) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_normal::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ClampScalarLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ClampScalarMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, ClampScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ClampScalarLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ClampScalarMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, ClampScalarPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ClampScalarLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ClampScalarMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, ClampScalar) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min          = 3.0f;
        constexpr float max          = 4.0f;

        nn::vector<float> result(COUNT);
        nn::_math_cuda::clamp(
                COUNT, data.data(nn::loc_type::device),
                min, max, result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp((float)data[i], min, max);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Min) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_normal::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MinLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MinMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MinPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MinLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MinMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MinPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MinLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MinMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Min) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::min(
                COUNT, v1.data(nn::loc_type::device),
                v2.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), v1.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Max) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_normal::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MaxLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, MaxMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MaxPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MaxLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, MaxMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MaxPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MaxLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, MaxMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::_m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Max) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::max(
                COUNT, v1.data(nn::loc_type::device),
                v2.data(nn::loc_type::device),
                result.data(nn::loc_type::device, true), v1.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, Clamp) {
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min[COUNT]  = { 2, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_normal::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampPrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min[COUNT]  = { 2, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ClampLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min[COUNT]  = { 1, 2, 1 };
        constexpr float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, ClampMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float min[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };
        constexpr float max[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, ClampPrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ClampLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min[COUNT]  = { 1, 2, 1 };
        constexpr float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, ClampMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, ClampPrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ClampLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min[COUNT]  = { 1, 2, 1 };
        constexpr float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, ClampMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::_m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, Clamp) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 16;

        const nn::vector<float> data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector<float> min  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        const nn::vector<float> max  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        nn::vector<float> result(COUNT);
        nn::_math_cuda::clamp(
                COUNT, data.data(nn::loc_type::device), min.data(nn::loc_type::device),
                max.data(nn::loc_type::device), result.data(nn::loc_type::device, true), data.stream());

        nn::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, CompareTrue) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4 };

        bool ans;
        nn::_math_normal::compare(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(MathTest, CompareFalse) {
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 1, 5, 3, 4 };

        bool ans;
        nn::_math_normal::compare(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(SSETest, CompareTruePrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4 };

        bool ans;
        nn::_math_simd::compare<simd::_m128>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, CompareFalsePrecise) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 1, 5, 3, 4 };

        bool ans;
        nn::_math_simd::compare<simd::_m128>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, CompareTrueLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math_simd::compare<simd::_m128>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, CompareFalseLess) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math_simd::compare<simd::_m128>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, CompareTrueMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 5;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5 };

        bool ans;
        nn::_math_simd::compare<simd::_m128>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(SSETest, CompareFalseMore) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        constexpr uint32_t COUNT = 5;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5 };
        constexpr float v2[COUNT] = { 1, 5, 3, 4, 5 };

        bool ans;
        nn::_math_simd::compare<simd::_m128>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, CompareTruePrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };

        bool ans;
        nn::_math_simd::compare<simd::_m256>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, CompareFalsePrecise) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8 };

        bool ans;
        nn::_math_simd::compare<simd::_m256>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, CompareTrueLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math_simd::compare<simd::_m256>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, CompareFalseLess) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math_simd::compare<simd::_m256>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, CompareTrueMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 9;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        bool ans;
        nn::_math_simd::compare<simd::_m256>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVXTest, CompareFalseMore) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        constexpr uint32_t COUNT = 9;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9 };

        bool ans;
        nn::_math_simd::compare<simd::_m256>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, CompareTruePrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        bool ans;
        nn::_math_simd::compare<simd::_m512>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, CompareFalsePrecise) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 14, 14, 15 };

        bool ans;
        nn::_math_simd::compare<simd::_m512>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, CompareTrueLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math_simd::compare<simd::_m512>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, CompareFalseLess) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math_simd::compare<simd::_m512>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, CompareTrueMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };

        bool ans;
        nn::_math_simd::compare<simd::_m512>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(AVX512Test, CompareFalseMore) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 20, 21 };

        bool ans;
        nn::_math_simd::compare<simd::_m512>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, CompareTrue) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 4;

        const nn::vector<float> first = { 1, 2, 3, 4 };
        const nn::vector<float> second = { 1, 2, 3, 4 };

        bool ans;
        nn::_math_cuda::compare(
                COUNT, first.data(nn::loc_type::device),
                second.data(nn::loc_type::device), &ans, first.stream());

        EXPECT_TRUE(ans);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(CudaTest, CompareFalse) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        constexpr uint32_t COUNT = 4;

        const nn::vector<float> first = { 1, 2, 3, 4 };
        const nn::vector<float> second = { 1, 5, 3, 4 };

        bool ans;
        nn::_math_cuda::compare(
                COUNT, first.data(nn::loc_type::device),
                second.data(nn::loc_type::device), &ans, first.stream());

        EXPECT_FALSE(ans);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(MathTest, MatrixMulRC) {
        const nn::matrix<float> first = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };

        const nn::matrix<float> second({
                { 2, 6, 0, 4, 7 },
                { 5, 7, 8, 3, 1 }
        }, nn::column_major);

        nn::matrix<float> result(second.width(), first.height());
        nn::_math_normal::matmul_rc(
                first.width(), first.height(), second.width(), first.data(nn::loc_type::host),
                second.data(nn::loc_type::host), result.data(nn::loc_type::host, true));

        EXPECT_EQ(result, nn::matrix<float>({{ 53, 59 }, { 89, 36 }, { 47, 69 }}));
}

TEST(SSETest, MatrixMulRC) {
#if SIMD_SUPPORT_LEVEL < 1
        GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 1
        const nn::matrix<float> first = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };

        const nn::matrix<float> second({
                { 2, 6, 0, 4, 7 },
                { 5, 7, 8, 3, 1 }
        }, nn::column_major);

        nn::matrix<float> result(second.width(), first.height());
        nn::_math_simd::matmul_rc<simd::_m128>(
                first.width(), first.height(), second.width(), first.data(nn::loc_type::host),
                second.data(nn::loc_type::host), result.data(nn::loc_type::host, true));

        EXPECT_EQ(result, nn::matrix<float>({{ 53, 59 }, { 89, 36 }, { 47, 69 }}));
#endif // SIMD_SUPPORT_LEVEL < 1
}

TEST(AVXTest, MatrixMulRC) {
#if SIMD_SUPPORT_LEVEL < 2
        GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 2
        const nn::matrix<float> first = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 1 }
        };

        const nn::matrix<float> second({
                { 2, 6, 0, 4, 7, 6, 1, 2, 9 },
                { 5, 7, 8, 3, 1, 8, 2, 1, 3 }
        }, nn::column_major);

        nn::matrix<float> result(second.width(), first.height());
        nn::_math_simd::matmul_rc<simd::_m256>(
                first.width(), first.height(), second.width(), first.data(nn::loc_type::host),
                second.data(nn::loc_type::host), result.data(nn::loc_type::host, true));

        EXPECT_EQ(result, nn::matrix<float>({{ 154, 119 }, { 194, 124 }, { 67, 94 }, { 87, 117 }}));
#endif // SIMD_SUPPORT_LEVEL < 2
}

TEST(AVX512Test, MatrixMulRC) {
#if SIMD_SUPPORT_LEVEL < 3
        GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
#else // SIMD_SUPPORT_LEVEL < 3
        const nn::matrix<float> first = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8, 1, 2, 1, 4, 1, 9, 1, 5, 3 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6, 1, 1, 8, 6, 3, 2, 8, 1, 1 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0, 9, 2, 1, 9, 5, 3, 2, 2, 8 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 1, 1, 1, 3, 0, 1, 8, 4, 3, 9 }
        };

        const nn::matrix<float> second({
                { 2, 6, 0, 4, 7, 6, 1, 2, 9, 7, 1, 1, 0, 1, 2, 1, 4, 3 },
                { 5, 7, 8, 3, 1, 8, 2, 1, 3, 6, 5, 0, 1, 3, 3, 2, 6, 7 }
        }, nn::column_major);

        nn::matrix<float> result(second.width(), first.height());
        nn::_math_simd::matmul_rc<simd::_m512>(
                first.width(), first.height(), second.width(), first.data(nn::loc_type::host),
                second.data(nn::loc_type::host), result.data(nn::loc_type::host, true));

        EXPECT_EQ(result, nn::matrix<float>({{ 213, 222 }, { 232, 185 }, { 178, 263 }, { 158, 244 }}));
#endif // SIMD_SUPPORT_LEVEL < 3
}

TEST(CudaTest, MatrixMulRC) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        const nn::matrix<float> first = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };

        const nn::matrix<float> second({
                { 2, 6, 0, 4, 7 },
                { 5, 7, 8, 3, 1 }
        }, nn::column_major);

        nn::matrix<float> result(second.width(), first.height());
        nn::_math_cuda::matmul_rc(
                first.width(), first.height(), second.width(), first.data(nn::loc_type::device),
                second.data(nn::loc_type::device), result.data(nn::loc_type::device, true), first.stream());

        EXPECT_EQ(result, nn::matrix<float>({{ 53, 59 }, { 89, 36 }, { 47, 69 }}));
#endif // !BUILD_CUDA_SUPPORT
}
