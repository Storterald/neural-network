#include <gtest/gtest.h>

#include <initializer_list>

#include <neural-network/utils/exceptions.h>
#include <neural-network/types/matrix.h>
#include <neural-network/base.h>

#include "printers.h"

TEST(MatrixTest, DefaultConstructorInitializesEmptyData) {
        const nn::matrix<float> m{};

        EXPECT_EQ(m.data(), nullptr);
        EXPECT_EQ(m.size(), 0);
}

TEST(MatrixTest, SizesConstructorAllocatesData) {
        const nn::matrix<float> m(10, 10);

        EXPECT_NE(m.data(), nullptr);
        EXPECT_EQ(m.size(), 100);
        EXPECT_EQ(m.width(), 10);
        EXPECT_EQ(m.height(), 10);
        EXPECT_NO_FATAL_FAILURE([[maybe_unused]] float value = m[9][9]);
}

TEST(MatrixTest, ConstructorWithInitializerListWorks) {
        const std::initializer_list<std::initializer_list<float>> values = {
                { 1, 2, 3 },
                { 4, 5, 6 }
        };
        const nn::matrix<float> m(values);

        EXPECT_NE(m.data(), nullptr);
        EXPECT_EQ(m.height(), values.size());
        EXPECT_EQ(m.width(), values.begin()->size());
        for (uint32_t i = 0; i < m.height(); ++i)
                for (uint32_t j = 0; j < m.width(); ++j)
                        EXPECT_EQ(m[i][j], std::data(std::data(values)[i])[j]);
}

TEST(MatrixTestDebug, ConstructorWithInitializerListThrowsIfSizesDontMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const std::initializer_list<std::initializer_list<float>> values = {
                { 1, 2, 3, 4 },
                { 5, 6, 7 }
        };
        EXPECT_THROW(nn::matrix<float> m(values), nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, AccessOperator) {
        nn::matrix<float> m(10, 10);
        m[9][9] = 3.0f;

        EXPECT_EQ(m[9][9], 3.0f);
}

TEST(MatrixTestDebug, AccessOperatorThrowsIfIndexOutOfBounds) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m(10, 10);
        EXPECT_THROW([[maybe_unused]] auto value = m[10], nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, PairAccessOperator) {
        nn::matrix<float> m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m[nn::indexer(9, 9)], 3.0f);
}

TEST(MatrixTest, PairAccessOperatorMatchesAccessOperator) {
        nn::matrix<float> m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m[nn::indexer(9, 9)], m[9][9]);
}

TEST(MatrixTestDebug, PairAccessOperatorThrowsIfRowIndexOutOfBounds) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m(10, 10);
        EXPECT_THROW([[maybe_unused]] float value = m[nn::indexer(10, 9)], nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTestDebug, PairAccessOperatorThrowsIfHeightIndexOutOfBounds) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m(10, 10);
        EXPECT_THROW([[maybe_unused]] float value = m[nn::indexer(9, 10)], nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, AtFunction) {
        nn::matrix<float> m(10, 10);
        m[9][9] = 3.0f;

        EXPECT_EQ(m.at(9)[9], 3.0f);
}

TEST(MatrixTestDebug, AtFunctionThrowsIfIndexOutOfBounds) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m(10, 10);
        EXPECT_THROW([[maybe_unused]] auto value = m.at(10), nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, PairAtFunctionReturnsCorrectValue) {
        nn::matrix<float> m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m.at(9, 9), 3.0f);
}

TEST(MatrixTest, PairAtFunctionMatchesAtFunction) {
        nn::matrix<float> m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m.at(9)[9], m.at(9, 9));
}

TEST(MatrixTestDebug, PairAtFunctionThrowsIfRowIndexOutOfBounds) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m(10, 10);
        EXPECT_THROW([[maybe_unused]] float value = m.at(10, 9), nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTestDebug, PairAtFunctionThrowsIfHeightIndexOutOfBounds) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m(10, 10);
        EXPECT_THROW([[maybe_unused]] float value = m.at(9, 10), nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, AdditionOperator) {
        const nn::matrix<float> m1     = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2     = { { 5, 6 }, { 7, 8 } };
        const nn::matrix<float> result = m1 + m2;
        EXPECT_EQ(result, nn::matrix<float>({ { 6, 8 }, { 10, 12 } }));
}

TEST(MatrixTestDebug, AdditionOperatorThrowsIfWidthDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m1 = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6, 7 }, { 8, 9, 10 } };
        EXPECT_THROW(nn::matrix value = m1 + m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTestDebug, AdditionOperatorThrowsIfHeightDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m1 = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6 }, { 7, 8 }, { 9, 10 } };
        EXPECT_THROW(nn::matrix value = m1 + m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, AdditionAssignmentOperator) {
        nn::matrix<float> m1       = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6 }, { 7, 8 } };
        m1                        += m2;
        EXPECT_EQ(m1, nn::matrix<float>({ { 6, 8 }, { 10, 12 } }));
}

TEST(MatrixTestDebug, AdditionAssignmentOperatorThrowsIfWidthDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        nn::matrix<float> m1       = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6, 7 }, { 8, 9, 10 } };
        EXPECT_THROW(m1 += m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTestDebug, AdditionAssignmentOperatorThrowsIfHeightDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        nn::matrix<float> m1       = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6 }, { 7, 8 }, { 9, 10 } };
        EXPECT_THROW(m1 += m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, ScalarAdditionOperator) {
        const nn::matrix<float> m      = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> result = m + 5.0f;
        EXPECT_EQ(result, nn::matrix<float>({ { 6, 7 }, { 8, 9 } }));
}

TEST(MatrixTest, ScalarAdditionAssignmentOperator) {
        nn::matrix<float> m = { { 1, 2 }, { 3, 4 } };
        m                  += 5;
        EXPECT_EQ(m, nn::matrix<float>({ { 6, 7 }, { 8, 9 } }));
}

TEST(MatrixTest, SubtractionOperator) {
        const nn::matrix<float> m1     = { { 5, 6 }, { 7, 8 } };
        const nn::matrix<float> m2     = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> result = m1 - m2;
        EXPECT_EQ(result, nn::matrix<float>({ { 4, 4 }, { 4, 4 } }));
}

TEST(MatrixTestDebug, SubtractionOperatorThrowsIfWidthDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m1 = { { 1, 2 }, { 3, 4 }};
        const nn::matrix<float> m2 = { { 5, 6, 7 }, { 8, 9, 10 }};
        EXPECT_THROW(nn::matrix value = m1 - m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTestDebug, SubtractionOperatorThrowsIfHeightDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m1 = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6 }, { 7, 8 }, { 9, 10 } };
        EXPECT_THROW(nn::matrix value = m1 - m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, SubtractionAssignmentOperator) {
        nn::matrix<float> m1       = { { 5, 6 }, { 7, 8 } };
        const nn::matrix<float> m2 = { { 1, 2 }, { 3, 4 } };
        m1                        -= m2;
        EXPECT_EQ(m1, nn::matrix<float>({ { 4, 4 }, { 4, 4 } }));
}

TEST(MatrixTestDebug, SubtractionAssignmentOperatorThrowsIfWidthDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        nn::matrix<float> m1       = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6, 7 }, { 8, 9, 10 } };
        EXPECT_THROW(m1 -= m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTestDebug, SubtractionAssignmentOperatorThrowsIfHeightDoesNotMatch) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        nn::matrix<float> m1       = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> m2 = { { 5, 6 }, { 7, 8 }, { 9, 10 } };
        EXPECT_THROW(m1 -= m2, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, ScalarSubtractionOperator) {
        const nn::matrix<float> m      = { { 5, 6 }, { 7, 8 } };
        const nn::matrix<float> result = m - 2.0f;
        EXPECT_EQ(result, nn::matrix<float>({ { 3, 4 }, { 5, 6 } }));
}

TEST(MatrixTest, ScalarSubtractionAssignmentOperator) {
        nn::matrix<float> m = { { 5, 6 }, { 7, 8 } };
        m                  -= 2;
        EXPECT_EQ(m, nn::matrix<float>({ { 3, 4 }, { 5, 6 } }));
}

TEST(MatrixTest, ScalarMultiplicationOperator) {
        const nn::matrix<float> m      = { { 1, 2 }, { 3, 4 } };
        const nn::matrix<float> result = m * 3.0f;
        EXPECT_EQ(result, nn::matrix<float>({ { 3, 6 }, { 9, 12 } }));
}

TEST(MatrixTest, ScalarMultiplicationAssignmentOperator) {
        nn::matrix<float> m = { { 1, 2 }, { 3, 4 } };
        m                  *= 3;
        EXPECT_EQ(m, nn::matrix<float>({ { 3, 6 }, { 9, 12 } }));
}

TEST(MatrixTest, ScalarDivisionOperator) {
        const nn::matrix<float> m      = { { 6, 8 }, { 10, 12 } };
        const nn::matrix<float> result = m / 2.0f;
        EXPECT_EQ(result, nn::matrix<float>({ { 3, 4 }, { 5, 6 } }));
}

TEST(MatrixTestDebug, ScalarDivisionOperatorThrowsIfValueIs0) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m = { { 6, 8 }, { 10, 12 } };
        EXPECT_THROW(nn::matrix value = m / 0.0f, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, ScalarDivisionAssignmentOperator) {
        nn::matrix<float> m = { { 6, 8 }, { 10, 12 } };
        m                  /= 2;
        EXPECT_EQ(m, nn::matrix<float>({ { 3, 4 }, { 5, 6 } }));
}

TEST(MatrixTestDebug, ScalarDivisionAssignmentOperatorThrowsIfValueIs0) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        nn::matrix<float> m = { { 6, 8 }, { 10, 12 } };
        EXPECT_THROW(m /= 0, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}

TEST(MatrixTest, VectorMultiplicationOperator) {
        const nn::matrix<float> m      = { { 1, 2, 1, 1 }, { 0, 1, 0, 1 }, { 2, 3, 4, 1 } };
        const nn::vector<float> v      = { 2, 6, 1, 1 };
        const nn::vector<float> result = m * v;
        EXPECT_EQ(result, nn::vector<float>({ 16, 7, 27 }));
}

TEST(MatrixTestDebug, VectorMultiplicationOperatorThrowsIfVectorSizeDoesNotMatchMatrixWidth) {
#ifndef DEBUG_MODE_ENABLED
        GTEST_SKIP() << "Skipping DEBUG tests in release mode.";
#else // !DEBUG_MODE_ENABLED
        const nn::matrix<float> m = { { 1, 2, 1, 1 }, { 0, 1, 0, 1 }, { 2, 3, 4, 1 } };
        const nn::vector<float> v = { 2, 6, 1 };
        EXPECT_THROW(nn::vector value = m * v, nn::fatal_error);
#endif // !DEBUG_MODE_ENABLED
}
