#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <cmath>

#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>
#include "../src/math/_math_normal.h"

#ifdef TARGET_X86_64
#include <neural-network/intrinsic/intrinsic.h>
#include <neural-network/utils/simd.h>
#include "../src/math/_math_simd.h"

namespace simd = nn::simd;
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
#include "../src/math/_math_cuda.h"
#endif // BUILD_CUDA_SUPPORT


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

#ifdef TARGET_X86_64
TEST(SSETest, SumPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Sum) {
        constexpr uint32_t COUNT = 16;

        const nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math_cuda::sum(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), v1.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] + v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, SubPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Sub) {
        constexpr uint32_t COUNT = 16;

        const nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math_cuda::sub(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), v1.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] - v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, MulPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Mul) {
        constexpr uint32_t COUNT = 16;

        const nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math_cuda::mul(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), v1.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] * v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, DivPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 8, 9, 10, 11, 12, 13, 14 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 5, 6, 7, 8, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 4, 5, 6 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Div) {
        constexpr uint32_t COUNT = 16;

        const nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector v2 = { 9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16 };

        nn::vector result(COUNT);
        nn::_math_cuda::div(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), v1.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = v1[i] / v2[i];

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, SumScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SumScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SumScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SumScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sum<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, SumScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::sum(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] + scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, SubScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, SubScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, SubScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, SubScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::sub<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, SubScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::sub(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] - scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, MulScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MulScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MulScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MulScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::mul<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MulScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::mul(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] * scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, DivScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, DivScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m128>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, DivScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m256>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, DivScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float scalar      = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::div<simd::m512>(COUNT, data, scalar, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, DivScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data  = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float scalar = 3.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::div(
                COUNT, data.view(nn::buf::DEVICE),
                scalar, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] / scalar;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, TanhPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Tanh) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math_cuda::tanh(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::tanh(data[i]);

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, TanhDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(SSETest, TanhDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                float tanh = std::tanh(data[i]);
                expected[i] = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVXTest, TanhDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}

TEST(AVX512Test, TanhDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::tanh_derivative<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, TanhDerivative) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math_cuda::tanh_derivative(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i) {
                const float tanh = std::tanh(data[i]);
                expected[i]      = 1 - tanh * tanh;
        }

        EXPECT_EQ_FLOAT_VEC(result, expected, 4);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, ReLUPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLULess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLULess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLULess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], 0.0f);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ReLU) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math_cuda::ReLU(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], 0.0f);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, ReLUDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ReLUDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m128>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ReLUDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m256>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativeLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ReLUDerivativeMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        std::vector<float> result(COUNT);
        nn::_math_simd::ReLU_derivative<simd::m512>(COUNT, data, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ReLUDerivative) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };

        nn::vector result(COUNT);
        nn::_math_cuda::ReLU_derivative(
                COUNT, data.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = data[i] >= 0.0f ? 1.0f : 0.0f;

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, MinScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m128>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m128>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m128>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m256>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m256>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m256>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m512>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m512>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m512>(COUNT, data, min, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(data[i], min);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MinScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min   = 3.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::min(
                COUNT, data.view(nn::buf::DEVICE),
                min, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min((float)data[i], min);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, MaxScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m128>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m128>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m128>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m256>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m256>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m256>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m512>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m512>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float max         = 3.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m512>(COUNT, data, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(data[i], max);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MaxScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float max   = 3.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::max(
                COUNT, data.view(nn::buf::DEVICE),
                max, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max((float)data[i], max);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, ClampScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampScalarMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min         = 3.0f;
        constexpr float max         = 4.0f;

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min, max);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, ClampScalar) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min   = 3.0f;
        constexpr float max   = 4.0f;

        nn::vector result(COUNT);
        nn::_math_cuda::clamp(
                COUNT, data.view(nn::buf::DEVICE),
                min, max, result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp((float)data[i], min, max);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, MinPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MinMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MinMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MinMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::min<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Min) {
        constexpr uint32_t COUNT = 16;

        const nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        nn::vector result(COUNT);
        nn::_math_cuda::min(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), v1.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::min(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, MaxPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, MaxMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float v2[COUNT] = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m128>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 4, 3, 2, 1, 1, 2, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, MaxMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m256>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, MaxMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float v2[COUNT] = { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::max<simd::m512>(COUNT, v1, v2, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Max) {
        constexpr uint32_t COUNT = 16;

        const nn::vector v1 = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector v2 = { 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1 };

        nn::vector result(COUNT);
        nn::_math_cuda::max(
                COUNT, v1.view(nn::buf::DEVICE),
                v2.view(nn::buf::DEVICE),
                result.data(nn::buf::DEVICE, true), v1.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::max(v1[i], v2[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, ClampPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float data[COUNT] = { 1, 2, 3, 4 };
        constexpr float min[COUNT]  = { 2, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min[COUNT]  = { 1, 2, 1 };
        constexpr float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(SSETest, ClampMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 7;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7 };
        constexpr float min[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };
        constexpr float max[COUNT]  = { 7, 6, 5, 4, 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m128>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 1, 2, 3, 4 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min[COUNT]  = { 1, 2, 1 };
        constexpr float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVXTest, ClampMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 11;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m256>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampPrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float data[COUNT] = { 1, 2, 3 };
        constexpr float min[COUNT]  = { 1, 2, 1 };
        constexpr float max[COUNT]  = { 3, 2, 1 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}

TEST(AVX512Test, ClampMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float data[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        constexpr float min[COUNT]  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1 };
        constexpr float max[COUNT]  = { 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4 };

        std::vector<float> result(COUNT);
        nn::_math_simd::clamp<simd::m512>(COUNT, data, min, max, result.data());

        std::vector<float> expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, Clamp) {
        constexpr uint32_t COUNT = 16;

        const nn::vector data = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
        const nn::vector min  = { 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1 };
        const nn::vector max  = { 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 3 };

        nn::vector result(COUNT);
        nn::_math_cuda::clamp(
                COUNT, data.view(nn::buf::DEVICE), min.view(nn::buf::DEVICE),
                max.view(nn::buf::DEVICE), result.data(nn::buf::DEVICE, true), data.stream());

        nn::vector expected(COUNT);
        for (uint32_t i = 0; i < COUNT; ++i)
                expected[i] = std::clamp(data[i], min[i], max[i]);

        EXPECT_EQ(result, expected);
}
#endif // BUILD_CUDA_SUPPORT

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

#ifdef TARGET_X86_64
TEST(SSETest, CompareTruePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4 };

        bool ans;
        nn::_math_simd::compare<simd::m128>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(SSETest, CompareFalsePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 4;

        constexpr float v1[COUNT] = { 1, 2, 3, 4 };
        constexpr float v2[COUNT] = { 1, 5, 3, 4 };

        bool ans;
        nn::_math_simd::compare<simd::m128>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(SSETest, CompareTrueLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math_simd::compare<simd::m128>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(SSETest, CompareFalseLess) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math_simd::compare<simd::m128>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(SSETest, CompareTrueMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 5;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5 };

        bool ans;
        nn::_math_simd::compare<simd::m128>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(SSETest, CompareFalseMore) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 5;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5 };
        constexpr float v2[COUNT] = { 1, 5, 3, 4, 5 };

        bool ans;
        nn::_math_simd::compare<simd::m128>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVXTest, CompareTruePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };

        bool ans;
        nn::_math_simd::compare<simd::m256>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVXTest, CompareFalsePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 8;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8 };

        bool ans;
        nn::_math_simd::compare<simd::m256>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVXTest, CompareTrueLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math_simd::compare<simd::m256>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVXTest, CompareFalseLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math_simd::compare<simd::m256>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVXTest, CompareTrueMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 9;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        bool ans;
        nn::_math_simd::compare<simd::m256>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVXTest, CompareFalseMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 9;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9 };

        bool ans;
        nn::_math_simd::compare<simd::m256>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVX512Test, CompareTruePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        bool ans;
        nn::_math_simd::compare<simd::m512>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVX512Test, CompareFalsePrecise) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 16;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 14, 14, 15 };

        bool ans;
        nn::_math_simd::compare<simd::m512>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVX512Test, CompareTrueLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 2, 3 };

        bool ans;
        nn::_math_simd::compare<simd::m512>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVX512Test, CompareFalseLess) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 3;

        constexpr float v1[COUNT] = { 1, 2, 3 };
        constexpr float v2[COUNT] = { 1, 5, 3 };

        bool ans;
        nn::_math_simd::compare<simd::m512>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}

TEST(AVX512Test, CompareTrueMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };

        bool ans;
        nn::_math_simd::compare<simd::m512>(COUNT, v1, v2, &ans);

        EXPECT_TRUE(ans);
}

TEST(AVX512Test, CompareFalseMore) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        constexpr uint32_t COUNT = 22;

        constexpr float v1[COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
        constexpr float v2[COUNT] = { 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 20, 21 };

        bool ans;
        nn::_math_simd::compare<simd::m512>(COUNT, v1, v2, &ans);

        EXPECT_FALSE(ans);
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, CompareTrue) {
        constexpr uint32_t COUNT = 4;

        const nn::vector first = { 1, 2, 3, 4 };
        const nn::vector second = { 1, 2, 3, 4 };

        bool ans;
        nn::_math_cuda::compare(
                COUNT, first.view(nn::buf::DEVICE),
                second.view(nn::buf::DEVICE), &ans, first.stream());

        EXPECT_TRUE(ans);
}

TEST(CudaTest, CompareFalse) {
        constexpr uint32_t COUNT = 4;

        const nn::vector first = { 1, 2, 3, 4 };
        const nn::vector second = { 1, 5, 3, 4 };

        bool ans;
        nn::_math_cuda::compare(
                COUNT, first.view(nn::buf::DEVICE),
                second.view(nn::buf::DEVICE), &ans, first.stream());

        EXPECT_FALSE(ans);
}
#endif // BUILD_CUDA_SUPPORT

TEST(MathTest, MatrixVectorMul) {
        const nn::matrix mat = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };
        const nn::vector vec = { 2, 6, 0, 4, 7 };

        nn::vector result(mat.height());
        nn::_math_normal::matvec_mul(
                mat.width(), mat.height(), mat.view(nn::buf::HOST),
                vec.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 53, 89, 47 }));
}

#ifdef TARGET_X86_64
TEST(SSETest, MatrixVectorMul) {
        if (nn::intrinsic::support() < nn::SIMD_SSE3) {
                GTEST_SKIP() << "Skipping SSE3 tests as it's not supported.";
                return;
        }

        const nn::matrix mat = {
                { 1, 2, 4, 1, 5 },
                { 0, 3, 0, 2, 9 },
                { 2, 3, 4, 1, 3 }
        };
        const nn::vector vec = { 2, 6, 0, 4, 7 };

        nn::vector result(mat.height());
        nn::_math_simd::matvec_mul<simd::m128>(
                mat.width(), mat.height(), mat.view(nn::buf::HOST),
                vec.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 53, 89, 47 }));
}

TEST(AVXTest, MatrixVectorMul) {
        if (nn::intrinsic::support() < nn::SIMD_AVX) {
                GTEST_SKIP() << "Skipping AVX tests as it's not supported.";
                return;
        }

        const nn::matrix mat = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 1 }
        };
        const nn::vector vec = { 2, 6, 0, 4, 7, 6, 1, 2, 9 };

        nn::vector result(mat.height());
        nn::_math_simd::matvec_mul<simd::m256>(
                mat.width(), mat.height(), mat.view(nn::buf::HOST),
                vec.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 154, 194, 67, 87 }));
}

TEST(AVX512Test, MatrixVectorMul) {
        if (nn::intrinsic::support() < nn::SIMD_AVX512) {
                GTEST_SKIP() << "Skipping AVX512 tests as it's not supported.";
                return;
        }

        const nn::matrix mat = {
                { 1, 2, 4, 1, 5, 4, 1, 2, 8, 1, 2, 1, 4, 1, 9, 1, 5, 3 },
                { 0, 3, 0, 2, 9, 8, 3, 0, 6, 1, 1, 8, 6, 3, 2, 8, 1, 1 },
                { 2, 3, 4, 2, 3, 1, 6, 2, 0, 1, 2, 1, 9, 5, 3, 2, 2, 8 },
                { 1, 6, 4, 1, 1, 3, 1, 5, 9, 1, 1, 3, 0, 1, 8, 4, 3, 9 }
        };
        const nn::vector vec = { 2, 6, 0, 4, 7, 6, 1, 2, 9, 7, 1, 1, 0, 1, 2, 1, 4, 3 };

        nn::vector result(mat.height());
        nn::_math_simd::matvec_mul<simd::m512>(
                mat.width(), mat.height(), mat.view(nn::buf::HOST),
                vec.view(nn::buf::HOST), result.data(nn::buf::HOST, true));

        EXPECT_EQ(result, nn::vector({ 213, 232, 122, 230 }));
}
#endif // TARGET_X86_64

#ifdef BUILD_CUDA_SUPPORT
TEST(CudaTest, MatrixVectorMul) {
        const nn::matrix mat = {
                { 1, 2, 1, 1 },
                { 0, 1, 0, 1 },
                { 2, 3, 4, 1 }
        };
        const nn::vector vec = { 2, 6, 1, 1 };

        nn::vector result(mat.height());
        nn::_math_cuda::matvec_mul(
                mat.width(), mat.height(), mat.view(nn::buf::DEVICE),
                vec.view(nn::buf::DEVICE), result.data(nn::buf::DEVICE, true), mat.stream());

        EXPECT_EQ(result, nn::vector({ 16, 7, 27 }));
}
#endif // BUILD_CUDA_SUPPORT
