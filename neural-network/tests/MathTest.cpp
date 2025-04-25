#include "gtest/gtest.h"

#include <neural-network/types/Matrix.h> // used for easier test of the matvec_mul function
#include <neural-network/types/Vector.h> // used as a device memory container
#include <neural-network/math/_Math.h>
#include <algorithm>
#include <vector>

#define EXPECT_EQ_FLOAT_VEC(expected, actual, thresh)           \
do {                                                            \
        for (uint32_t idx = 0; idx < expected.size(); ++idx)    \
            EXPECT_NEAR(expected[idx], actual[idx], thresh);    \
} while (false)

TEST(MathTest, Sum) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_NORMAL>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumPrecise) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumMore) {
        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumPrecise) {
        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumMore) {
        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumPrecise) {
        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumMore) {
        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sum(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Sum) {
        constexpr uint32_t COUNT = 16;

        Vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::sum(
                COUNT, v1.as_span(Data::DEVICE, true),
                v2.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubPrecise) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubMore) {
        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubPrecise) {
        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubMore) {
        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubPrecise) {
        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubMore) {
        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sub(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Sub) {
        constexpr uint32_t COUNT = 16;

        Vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::sub(
                COUNT, v1.as_span(Data::DEVICE, true),
                v2.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulPrecise) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulMore) {
        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulPrecise) {
        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulMore) {
        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulPrecise) {
        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulMore) {
        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::mul(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Mul) {
        constexpr uint32_t COUNT = 16;

        Vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::mul(
                COUNT, v1.as_span(Data::DEVICE, true),
                v2.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivPrecise) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivMore) {
        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivPrecise) {
        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivMore) {
        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivPrecise) {
        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivMore) {
        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::div(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Div) {
        constexpr uint32_t COUNT = 16;

        Vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::div(
                COUNT, v1.as_span(Data::DEVICE, true),
                v2.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar       = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sum(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, SumScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::sum(
                COUNT, data.as_span(Data::DEVICE, true),
                scalar, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar    = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::sub(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, SubScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::sub(
                COUNT, data.as_span(Data::DEVICE, true),
                scalar, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::mul(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MulScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::mul(
                COUNT, data.as_span(Data::DEVICE, true),
                scalar, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::div(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, DivScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float scalar = 3.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::div(
                COUNT, data.as_span(Data::DEVICE, true),
                scalar, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, Tanh) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_NORMAL>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::tanh(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Tanh) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::tanh(
                COUNT, data.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, TanhDerivative) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_NORMAL>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativePrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativeLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativeMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativePrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativeLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativeMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativePrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativeLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativeMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::tanh_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, TanhDerivative) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::tanh_derivative(
                COUNT, data.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLULess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLULess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLULess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::ReLU(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ReLU) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::ReLU(
                COUNT, data.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], 0.0f);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, ReLUDerivative) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_NORMAL>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativePrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativeLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativeMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativePrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativeLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativeMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativePrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativeLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativeMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::ReLU_derivative(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ReLUDerivative) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::ReLU_derivative(
                COUNT, data.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::min(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MinScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min   = 3.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::min(
                COUNT, data.as_span(Data::DEVICE, true),
                min, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float max         = 3.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::max(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MaxScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float max   = 3.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::max(
                COUNT, data.as_span(Data::DEVICE, true),
                max, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min         = 3.0f;
        float max         = 4.0f;

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ClampScalar) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min   = 3.0f;
        float max   = 4.0f;

        Vector result(COUNT);
        _Math<MATH_CUDA>::clamp(
                COUNT, data.as_span(Data::DEVICE, true),
                min, max, result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinPrecise) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinMore) {
        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinPrecise) {
        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinMore) {
        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinPrecise) {
        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinMore) {
        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::min(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Min) {
        constexpr uint32_t COUNT = 16;

        Vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::min(
                COUNT, v1.as_span(Data::DEVICE, true),
                v2.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxPrecise) {
        constexpr uint32_t COUNT = 4;

        float v1[COUNT] = { 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxMore) {
        constexpr uint32_t COUNT = 7;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxPrecise) {
        constexpr uint32_t COUNT = 8;

        float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxMore) {
        constexpr uint32_t COUNT = 11;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxPrecise) {
        constexpr uint32_t COUNT = 16;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxLess) {
        constexpr uint32_t COUNT = 3;

        float v1[COUNT] = { 1, 2, 3 };
        float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxMore) {
        constexpr uint32_t COUNT = 22;

        float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::max(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Max) {
        constexpr uint32_t COUNT = 16;

        Vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::max(
                COUNT, v1.as_span(Data::DEVICE, true),
                v2.as_span(Data::DEVICE, true),
                result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
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
        _Math<MATH_NORMAL>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampPrecise) {
        constexpr uint32_t COUNT = 4;

        float data[COUNT] = { 1, 2, 3, 4 };
        float min[COUNT]  = { 2, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min[COUNT]  = { 1, 2, 1 };
        float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampMore) {
        constexpr uint32_t COUNT = 7;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        float min[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };
        float max[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_SSE>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampPrecise) {
        constexpr uint32_t COUNT = 8;

        float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min[COUNT]  = { 1, 2, 1 };
        float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampMore) {
        constexpr uint32_t COUNT = 11;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampPrecise) {
        constexpr uint32_t COUNT = 16;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampLess) {
        constexpr uint32_t COUNT = 3;

        float data[COUNT] = { 1, 2, 3 };
        float min[COUNT]  = { 1, 2, 1 };
        float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampMore) {
        constexpr uint32_t COUNT = 22;

        float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        _Math<MATH_AVX512>::clamp(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Clamp) {
        constexpr uint32_t COUNT = 16;

        Vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        Vector min  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        Vector max  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        Vector result(COUNT);
        _Math<MATH_CUDA>::clamp(
                COUNT, data.as_span(Data::DEVICE, true), min.as_span(Data::DEVICE, true),
                max.as_span(Data::DEVICE, true), result.as_span(Data::DEVICE, true));

        Vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, MatrixVectorMul) {
        Matrix m = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };
        Vector v = { 2, 6, 0, 4, 7 };

        Vector result(m.height());
        _Math<MATH_NORMAL>::matvec_mul(
                m.width(), m.height(), m.as_span(Data::HOST, true),
                v.as_span(Data::HOST, true), result.as_span(Data::HOST, true));

        EXPECT_EQ(result, Vector({ 53, 89, 47 }));
}

TEST(SSETest, MatrixVectorMul) {
        Matrix m = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };
        Vector v = { 2, 6, 0, 4, 7 };

        Vector result(m.height());
        _Math<MATH_SSE>::matvec_mul(
                m.width(), m.height(), m.as_span(Data::HOST, true),
                v.as_span(Data::HOST, true), result.as_span(Data::HOST, true));

        EXPECT_EQ(result, Vector({ 53, 89, 47 }));
}

TEST(AVXTest, MatrixVectorMul) {
        Matrix m = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 1 }
        };
        Vector v = { 2, 6, 0, 4, 7, 6, 1, 2, 9 };

        Vector result(m.height());
        _Math<MATH_AVX>::matvec_mul(
                m.width(), m.height(), m.as_span(Data::HOST, true),
                v.as_span(Data::HOST, true), result.as_span(Data::HOST, true));

        EXPECT_EQ(result, Vector({ 154, 194, 67, 87 }));
}

TEST(AVX512Test, MatrixVectorMul) {
        Matrix m = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8, 1, 2, 1, 4, 1, 9, 1, 5, 3 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6, 1, 1, 8, 6, 3, 2, 8, 1, 1 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0, 1, 2, 1, 9, 5, 3, 2, 2, 8 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 9, 1, 1, 3, 0, 1, 8, 4, 3, 9 }
        };
        Vector v = { 2, 6, 0, 4, 7, 6, 1, 2, 9, 7, 1, 1, 0, 1, 2, 1, 4, 3 };

        Vector result(m.height());
        _Math<MATH_AVX512>::matvec_mul(
                m.width(), m.height(), m.as_span(Data::HOST, true),
                v.as_span(Data::HOST, true), result.as_span(Data::HOST, true));

        EXPECT_EQ(result, Vector({ 213, 232, 122, 230 }));
}

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MatrixVectorMul) {
        Matrix m = { { 1, 2, 1, 1 }, { 0, 1, 0, 1 }, { 2, 3, 4, 1 } };
        Vector v = { 2, 6, 1, 1 };

        Vector result(m.height());
        _Math<MATH_CUDA>::matvec_mul(
                m.width(), m.height(), m.as_span(Data::DEVICE, true),
                v.as_span(Data::DEVICE, true), result.as_span(Data::DEVICE, true));

        EXPECT_EQ(result, Vector({ 16, 7, 27 }));
}
#endif // BUILD_CUDA_SUPPORT
