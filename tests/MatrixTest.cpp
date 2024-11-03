#include <gtest/gtest.h>

#include "CUDATest.h"
#include "../src/types/Matrix.h"
#include "../src/utils/Logger.h"

TEST(MatrixTest, DefaultConstructorInitializesEmptyData) {
        Matrix m{};

        EXPECT_EQ(m.data(), nullptr);
        EXPECT_EQ(m.size(), 0);
}

TEST(MatrixTest, SizesConstuctorAllocatesData) {
        Matrix m(10, 10);

        EXPECT_NE(m.data(), nullptr);
        EXPECT_EQ(m.size(), 100);
        EXPECT_EQ(m.width(), 10);
        EXPECT_EQ(m.height(), 10);
        EXPECT_NO_FATAL_FAILURE(float value = m[9][9]);
}

TEST(MatrixTest, MatrixDataIsAccessibleFromGPU) {
        Matrix m(10, 10);
        for (int i{}; i < 100; ++i)
                m.data()[i] = 1;

        EXPECT_NE(m.data(), nullptr);
        EXPECT_EQ(m.size(), 100);
        EXPECT_EQ(m.width(), 10);
        EXPECT_EQ(m.height(), 10);
        EXPECT_NO_FATAL_FAILURE(CUDATest::accessValues(m));
        EXPECT_TRUE(CUDATest::checkValues(m, 1));
}

TEST(MatrixTest, ConstructorWithInitializerListWorks) {
        std::initializer_list<std::initializer_list<float>> values = {{1, 2, 3}, {4, 5, 6}};
        Matrix m(values);

        EXPECT_NE(m.data(), nullptr);
        EXPECT_EQ(m.height(), values.size());
        EXPECT_EQ(m.width(), values.begin()->size());
        EXPECT_TRUE([&]() -> bool {
                bool value { true };
                for (int i{}; i < m.height(); ++i)
                        for (int j{}; j < m.width(); ++j)
                                value &= m[i][j] == values.begin()[i].begin()[j];

                return value;
        }());
}

TEST(MatrixTest, ConstructorWithInitializerListThrowsIfSizesDontMatch) {
        std::initializer_list<std::initializer_list<float>> values = {{1, 2, 3, 4}, {5, 6, 7}};
        EXPECT_THROW(Matrix m(values), Logger::fatal_error);
}

TEST(MatrixTest, AccessOperator) {
        Matrix m(10, 10);
        m[9][9] = 3.0f;

        EXPECT_EQ(m[9][9], 3.0f);
}

TEST(MatrixTestDebug, AccessOperatorThrowsIfIndexOutOfBounds) {
        Matrix m(10, 10);
        EXPECT_THROW(float *value = m[10], Logger::fatal_error);
}

TEST(MatrixTest, PairAccessOperator) {
        Matrix m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m[std::pair(9, 9)], 3.0f);
}

TEST(MatrixTest, PairAccessOperatorMatchesAccessOperator) {
        Matrix m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m[std::pair(9, 9)], m[9][9]);
}

TEST(MatrixTestDebug, PairAccessOperatorThrowsIfRowIndexOutOfBounds) {
        Matrix m(10, 10);
        EXPECT_THROW(float value = m[std::pair(10, 9)], Logger::fatal_error);
}

TEST(MatrixTestDebug, PairAccessOperatorThrowsIfHeightIndexOutOfBounds) {
        Matrix m(10, 10);
        EXPECT_THROW(float value = m[std::pair(9, 10)], Logger::fatal_error);
}

TEST(MatrixTest, AtFunction) {
        Matrix m(10, 10);
        m[9][9] = 3.0f;

        EXPECT_EQ(m.at(9)[9], 3.0f);
}

TEST(MatrixTestDebug, AtFunctionThrowsIfIndexOutOfBounds) {
        Matrix m(10, 10);
        EXPECT_THROW(const float *value = m.at(10), Logger::fatal_error);
}

TEST(MatrixTest, PairAtFunction) {
        Matrix m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m.at(9, 9), 3.0f);
}

TEST(MatrixTest, PairAtFunctionMatchesAtFunction) {
        Matrix m(10, 10);
        m[{9, 9}] = 3.0f;

        EXPECT_EQ(m.at(9)[9], m.at(9, 9));
}

TEST(MatrixTestDebug, PairAtFunctionThrowsIfRowIndexOutOfBounds) {
        Matrix m(10, 10);
        EXPECT_THROW(float value = m.at(10, 9), Logger::fatal_error);
}

TEST(MatrixTestDebug, PairAtFunctionThrowsIfHeightIndexOutOfBounds) {
        Matrix m(10, 10);
        EXPECT_THROW(float value = m.at(9, 10), Logger::fatal_error);
}

TEST(MatrixTest, AdditionOperator) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6}, {7, 8}};
        Matrix result = m1 + m2;
        EXPECT_EQ(result, Matrix({{6, 8}, {10, 12}}));
}

TEST(MatrixTestDebug, AdditionOperatorThrowsIfWidthDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6, 7}, {8, 9, 10}};
        EXPECT_THROW(Matrix value = m1 + m2, Logger::fatal_error);
}

TEST(MatrixTestDebug, AdditionOperatorThrowsIfHeightDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6}, {7, 8}, {9, 10}};
        EXPECT_THROW(Matrix value = m1 + m2, Logger::fatal_error);
}

TEST(MatrixTest, AdditionAssignmentOperator) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6}, {7, 8}};
        m1 += m2;
        EXPECT_EQ(m1, Matrix({{6, 8}, {10, 12}}));
}

TEST(MatrixTestDebug, AdditionAssignmentOperatorThrowsIfWidthDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6, 7}, {8, 9, 10}};
        EXPECT_THROW(m1 += m2, Logger::fatal_error);
}

TEST(MatrixTestDebug, AdditionAssignmentOperatorThrowsIfHeightDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6}, {7, 8}, {9, 10}};
        EXPECT_THROW(m1 += m2, Logger::fatal_error);
}

TEST(MatrixTest, ScalarAdditionOperator) {
        Matrix m{{1, 2}, {3, 4}};
        Matrix result = m + 5;
        EXPECT_EQ(result, Matrix({{6, 7}, {8, 9}}));
}

TEST(MatrixTest, ScalarAdditionAssignmentOperator) {
        Matrix m{{1, 2}, {3, 4}};
        m += 5;
        EXPECT_EQ(m, Matrix({{6, 7}, {8, 9}}));
}

TEST(MatrixTest, SubtractionOperator) {
        Matrix m1{{5, 6}, {7, 8}};
        Matrix m2{{1, 2}, {3, 4}};
        Matrix result = m1 - m2;
        EXPECT_EQ(result, Matrix({{4, 4}, {4, 4}}));
}

TEST(MatrixTestDebug, SubtractionOperatorThrowsIfWidthDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6, 7}, {8, 9, 10}};
        EXPECT_THROW(Matrix value = m1 - m2, Logger::fatal_error);
}

TEST(MatrixTestDebug, SubtractionOperatorThrowsIfHeightDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6}, {7, 8}, {9, 10}};
        EXPECT_THROW(Matrix value = m1 - m2, Logger::fatal_error);
}

TEST(MatrixTest, SubtractionAssignmentOperator) {
        Matrix m1{{5, 6}, {7, 8}};
        Matrix m2{{1, 2}, {3, 4}};
        m1 -= m2;
        EXPECT_EQ(m1, Matrix({{4, 4}, {4, 4}}));
}

TEST(MatrixTestDebug, SubtractionAssignmentOperatorThrowsIfWidthDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6, 7}, {8, 9, 10}};
        EXPECT_THROW(m1 -= m2, Logger::fatal_error);
}

TEST(MatrixTestDebug, SubtractionAssignmentOperatorThrowsIfHeightDoesNotMatch) {
        Matrix m1{{1, 2}, {3, 4}};
        Matrix m2{{5, 6}, {7, 8}, {9, 10}};
        EXPECT_THROW(m1 -= m2, Logger::fatal_error);
}

TEST(MatrixTest, ScalarSubtractionOperator) {
        Matrix m{{5, 6}, {7, 8}};
        Matrix result = m - 2;
        EXPECT_EQ(result, Matrix({{3, 4}, {5, 6}}));
}

TEST(MatrixTest, ScalarSubtractionAssignmentOperator) {
        Matrix m{{5, 6}, {7, 8}};
        m -= 2;
        EXPECT_EQ(m, Matrix({{3, 4}, {5, 6}}));
}

TEST(MatrixTest, ScalarMultiplicationOperator) {
        Matrix m{{1, 2}, {3, 4}};
        Matrix result = m * 3;
        EXPECT_EQ(result, Matrix({{3, 6}, {9, 12}}));
}

TEST(MatrixTest, ScalarMultiplicationAssignmentOperator) {
        Matrix m{{1, 2}, {3, 4}};
        m *= 3;
        EXPECT_EQ(m, Matrix({{3, 6}, {9, 12}}));
}

TEST(MatrixTest, ScalarDivisionOperator) {
        Matrix m{{6, 8}, {10, 12}};
        Matrix result = m / 2;
        EXPECT_EQ(result, Matrix({{3, 4}, {5, 6}}));
}

TEST(MatrixTestDebug, ScalarDivisionOperatorThrowsIfValueIs0) {
        Matrix m{{6, 8}, {10, 12}};
        EXPECT_THROW(Matrix value = m / 0, Logger::fatal_error);
}

TEST(MatrixTest, ScalarDivisionAssignmentOperator) {
        Matrix m{{6, 8}, {10, 12}};
        m /= 2;
        EXPECT_EQ(m, Matrix({{3, 4}, {5, 6}}));
}

TEST(MatrixTestDebug, ScalarDivisionAssignmentOperatorThrowsIfValueIs0) {
        Matrix m{{6, 8}, {10, 12}};
        EXPECT_THROW(m /= 0, Logger::fatal_error);
}

TEST(MatrixTest, VectorMultiplicationOperator) {
        Matrix m{{1, 2, 1, 1}, {0, 1, 0, 1}, {2, 3, 4, 1}};
        Vector v{2, 6, 1, 1};
        Vector result = m * v;
        EXPECT_EQ(result, Vector({16, 7, 27}));
}

TEST(MatrixTestDebug, VectorMultiplicationOperatorThrowsIfVectorSizeDoesNotMatchMatrixWidth) {
        Matrix m{{1, 2, 1, 1}, {0, 1, 0, 1}, {2, 3, 4, 1}};
        Vector v{2, 6, 1};
        EXPECT_THROW(Vector value = m * v, Logger::fatal_error);
}
