#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <cmath>

#include <neural-network/intrinsic/intrinsic.h>
#include <neural-network/types/matrix.h> // used for easier test of the matvec_mul function
#include <neural-network/types/vector.h> // used as a device memory container
#include "../src/math/_math.h"

#define EXPECT_EQ_FLOAT_VEC(expected, actual, thresh)                   \
do {                                                                    \
        for (uint32_t idx = 0; idx < expected.size(); ++idx)            \
                EXPECT_NEAR(expected[idx], actual[idx], thresh);        \
} while (false)

TEST(MathTest, Sum) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, SumPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Sum) {
        constexpr uint32_t COUNT = 16;

        nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::sum(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Sub) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, SubPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Sub) {
        constexpr uint32_t COUNT = 16;

        nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::sub(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Mul) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, MulPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Mul) {
        constexpr uint32_t COUNT = 16;

        nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::mul(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Div) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, DivPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Div) {
        constexpr uint32_t COUNT = 16;

        nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::div(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, SumScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, SumScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar       = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, SumScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::sum(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, SubScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, SubScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar    = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, SubScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::sub(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, MulScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, MulScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MulScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::mul(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, DivScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, DivScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, DivScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::div(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Tanh) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

#ifdef IS_X86_64BIT
TEST(SSETest, TanhPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Tanh) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::tanh(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, TanhDerivative) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

#ifdef IS_X86_64BIT
TEST(SSETest, TanhDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, TanhDerivative) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::tanh_derivative(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, ReLU) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, ReLUPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLULess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLULess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLULess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ReLU) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::ReLU(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], 0.0f);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, ReLUDerivative) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, ReLUDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ReLUDerivative) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::ReLU_derivative(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, MinScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, MinScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MinScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min   = 3.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::min(
                COUNT, data.view(nn::buf::DEVICE),
                min, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min((float)data[i], min);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, MaxScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, MaxScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MaxScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float max   = 3.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::max(
                COUNT, data.view(nn::buf::DEVICE),
                max, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], max);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, ClampScalar) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, ClampScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ClampScalar) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min   = 3.0f;
        float max   = 4.0f;

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::clamp(
                COUNT, data.view(nn::buf::DEVICE),
                min, max, result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp((float)data[i], min, max);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Min) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, MinPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Min) {
        constexpr uint32_t COUNT = 16;

        nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::min(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Max) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, MaxPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Max) {
        constexpr uint32_t COUNT = 16;

        nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::max(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Clamp) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min[COUNT]  = { 2, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_NORMAL>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

#ifdef IS_X86_64BIT
TEST(SSETest, ClampPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min[COUNT]  = { 2, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min[COUNT]  = { 1, 2, 1 };
        float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float min[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };
        float max[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_SSE3>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min[COUNT]  = { 1, 2, 1 };
        float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min[COUNT]  = { 1, 2, 1 };
        float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math<nn::MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Clamp) {
        constexpr uint32_t COUNT = 16;

        nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        nn::vector min  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        nn::vector max  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        nn::vector result(COUNT);
        nn::_math<nn::MATH_CUDA>::clamp(
                COUNT, data.view(nn::buf::DEVICE), min.view(nn::buf::DEVICE),
                max.view(nn::buf::DEVICE), result.data(nn::buf::DEVICE, true), 0);

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, CompareTrue) {
        constexpr uint32_t COUNT = 4;

        float first[COUNT] = { 1, 2, 3, 4 };
        float second[COUNT] = { 1, 2, 3, 4 };

        bool ans;
        nn::_math<nn::MATH_NORMAL>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(MathTest, CompareFalse) {
        constexpr uint32_t COUNT = 4;

        float first[COUNT] = { 1, 2, 3, 4 };
        float second[COUNT] = { 1, 5, 3, 4 };

        bool ans;
        nn::_math<nn::MATH_NORMAL>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

#ifdef IS_X86_64BIT
TEST(SSETest, CompareTruePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float first[COUNT] = { 1, 2, 3, 4 };
        float second[COUNT] = { 1, 2, 3, 4 };

        bool ans;
        nn::_math<nn::MATH_SSE3>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(SSETest, CompareFalsePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 4;

        float first[COUNT] = { 1, 2, 3, 4 };
        float second[COUNT] = { 1, 5, 3, 4 };

        bool ans;
        nn::_math<nn::MATH_SSE3>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(SSETest, CompareTrueLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float first[COUNT] = { 1, 2, 3 };
        float second[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math<nn::MATH_SSE3>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(SSETest, CompareFalseLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 3;

        float first[COUNT] = { 1, 2, 3 };
        float second[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math<nn::MATH_SSE3>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(SSETest, CompareTrueMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 5;

        float first[COUNT] = { 1, 2, 3, 4, 5 };
        float second[COUNT] = { 1, 2, 3, 4, 5 };

        bool ans;
        nn::_math<nn::MATH_SSE3>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(SSETest, CompareFalseMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        constexpr uint32_t COUNT = 5;

        float first[COUNT] = { 1, 2, 3, 4, 5 };
        float second[COUNT] = { 1, 5, 3, 4, 5 };

        bool ans;
        nn::_math<nn::MATH_SSE3>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVXTest, CompareTruePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };

        bool ans;
        nn::_math<nn::MATH_AVX>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVXTest, CompareFalsePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 8;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8 };

        bool ans;
        nn::_math<nn::MATH_AVX>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVXTest, CompareTrueLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float first[COUNT] = { 1, 2, 3 };
        float second[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math<nn::MATH_AVX>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVXTest, CompareFalseLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 3;

        float first[COUNT] = { 1, 2, 3 };
        float second[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math<nn::MATH_AVX>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVXTest, CompareTrueMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 9;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        bool ans;
        nn::_math<nn::MATH_AVX>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVXTest, CompareFalseMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        constexpr uint32_t COUNT = 9;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9 };

        bool ans;
        nn::_math<nn::MATH_AVX>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVX512Test, CompareTruePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        bool ans;
        nn::_math<nn::MATH_AVX512>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVX512Test, CompareFalsePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 16;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 14, 14, 15 };

        bool ans;
        nn::_math<nn::MATH_AVX512>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVX512Test, CompareTrueLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float first[COUNT] = { 1, 2, 3 };
        float second[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math<nn::MATH_AVX512>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVX512Test, CompareFalseLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 3;

        float first[COUNT] = { 1, 2, 3 };
        float second[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math<nn::MATH_AVX512>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVX512Test, CompareTrueMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };

        bool ans;
        nn::_math<nn::MATH_AVX512>::compare(COUNT, first, second, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVX512Test, CompareFalseMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        constexpr uint32_t COUNT = 22;

        float first[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
        float second[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 20, 21 };

        bool ans;
        nn::_math<nn::MATH_AVX512>::compare(COUNT, first, second, &ans);

        EXPECT_FALSE(ans);
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, CompareTrue) {
        constexpr uint32_t COUNT = 4;

        nn::vector first = { 1, 2, 3, 4 };
        nn::vector second = { 1, 2, 3, 4 };

        bool ans;
        nn::_math<nn::MATH_CUDA>::compare(
                COUNT, first.view(nn::buf::DEVICE),
                second.view(nn::buf::DEVICE), &ans, 0);

        EXPECT_TRUE(ans);
}

TEST(CudaTest, CompareFalse) {
        constexpr uint32_t COUNT = 4;

        nn::vector first = { 1, 2, 3, 4 };
        nn::vector second = { 1, 5, 3, 4 };

        bool ans;
        nn::_math<nn::MATH_CUDA>::compare(
                COUNT, first.view(nn::buf::DEVICE),
                second.view(nn::buf::DEVICE), &ans, 0);

        EXPECT_FALSE(ans);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, MatrixVectorMul) {
        nn::matrix m = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };
        nn::vector v = { 2, 6, 0, 4, 7 };

        nn::vector result(m.height());
        nn::_math<nn::MATH_NORMAL>::matvec_mul(
                m.width(), m.height(), m.view(nn::buf::HOST),
                v.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 53, 89, 47 }));
}

#ifdef IS_X86_64BIT
TEST(SSETest, MatrixVectorMul) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3)
                return;

        nn::matrix m = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };
        nn::vector v = { 2, 6, 0, 4, 7 };

        nn::vector result(m.height());
        nn::_math<nn::MATH_SSE3>::matvec_mul(
                m.width(), m.height(), m.view(nn::buf::HOST),
                v.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 53, 89, 47 }));
}

TEST(AVXTest, MatrixVectorMul) {
        if (nn::intrinsic::support() < nn::SIMD_AVX)
                return;

        nn::matrix m = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 1 }
        };
        nn::vector v = { 2, 6, 0, 4, 7, 6, 1, 2, 9 };

        nn::vector result(m.height());
        nn::_math<nn::MATH_AVX>::matvec_mul(
                m.width(), m.height(), m.view(nn::buf::HOST),
                v.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 154, 194, 67, 87 }));
}

TEST(AVX512Test, MatrixVectorMul) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512)
                return;

        nn::matrix m = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8, 1, 2, 1, 4, 1, 9, 1, 5, 3 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6, 1, 1, 8, 6, 3, 2, 8, 1, 1 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0, 1, 2, 1, 9, 5, 3, 2, 2, 8 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 9, 1, 1, 3, 0, 1, 8, 4, 3, 9 }
        };
        nn::vector v = { 2, 6, 0, 4, 7, 6, 1, 2, 9, 7, 1, 1, 0, 1, 2, 1, 4, 3 };

        nn::vector result(m.height());
        nn::_math<nn::MATH_AVX512>::matvec_mul(
                m.width(), m.height(), m.view(nn::buf::HOST),
                v.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 213, 232, 122, 230 }));
}
#endif // IS_X86_64BIT

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MatrixVectorMul) {
        nn::matrix m = { { 1, 2, 1, 1 }, { 0, 1, 0, 1 }, { 2, 3, 4, 1 } };
        nn::vector v = { 2, 6, 1, 1 };

        nn::vector result(m.height());
        nn::_math<nn::MATH_CUDA>::matvec_mul(
                m.width(), m.height(), m.view(nn::buf::DEVICE),
                v.view(nn::buf::DEVICE), result.data(nn::buf::DEVICE, true), 0);

        EXPECT_EQ(result, nn::vector({ 16, 7, 27 }));
}
#endif // BUILD_CUDA_SUPPORT
