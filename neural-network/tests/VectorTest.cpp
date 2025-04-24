#include "gtest/gtest.h"

#include <neural-network/types/Vector.h>
#include <neural-network/utils/Logger.h>

#ifdef BUILD_CUDA_SUPPORT
#include "CudaTestHelper.h"
#endif // BUILD_CUDA_SUPPORT

TEST(VectorTest, DefaultConstructorInitializesEmptyData) {
        Vector v{};

        EXPECT_EQ(v.data(), nullptr);
        EXPECT_EQ(v.size(), 0);
}

TEST(VectorTest, SizeConstructorAllocatesData) {
        Vector v(10);

        EXPECT_NE(v.data(), nullptr);
        EXPECT_EQ(v.size(), 10);
        EXPECT_NO_FATAL_FAILURE(float value = v[9]);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(VectorTest, VectorDataIsAccessibleFromGPU) {
        Vector v(10);
        for (uint32_t i = 0; i < 10; ++i)
                v[i] = 1;

        EXPECT_NE(v.data(), nullptr);
        EXPECT_EQ(v.size(), 10);
        EXPECT_NO_FATAL_FAILURE(Helper::access_values(v));
        EXPECT_TRUE(Helper::check_values(v, 1));
}
#endif // BUILD_CUDA_SUPPORT

TEST(VectorTest, ConstructorWithFloatArrayWorks) {
        float values[] = { 1.0f, 2.0f, 3.0f };
        Vector v(3, values);

        EXPECT_NE(v.data(), nullptr);
        EXPECT_EQ(v.size(), 3);
        EXPECT_TRUE([&]() -> bool {
                bool value = true;
                for (uint32_t i = 0; i < 3; ++i)
                        value &= v[i] == values[i];

                return value;
        }());
}

TEST(VectorTest, ConstructorWithInitializerListWorks) {
        auto values = { 1.0f, 2.0f, 3.0f };
        Vector v(values);

        EXPECT_NE(v.data(), nullptr);
        EXPECT_EQ(v.size(), values.size());
        EXPECT_TRUE([&]() -> bool {
                bool value = true;
                for (uint32_t i = 0; i < values.size(); ++i)
                        value &= v[i] == values.begin()[i];

                return value;
        }());
}

TEST(VectorTest, AccessOperator) {
        Vector v = { 1.0f, 2.0f, 3.0f };

        EXPECT_EQ(v[2], 3.0f);
}

TEST(VectorTestDebug, AccessOperatorThrowsIfIndexOutOfBounds) {
        Vector v(10);
        EXPECT_THROW(float value = v[10], Logger::fatal_error);
}

TEST(VectorTest, AtFunction) {
        Vector v = { 1.0f, 2.0f, 3.0f };

        EXPECT_EQ(v.at(2), 3.0f);
}

TEST(VectorTestDebug, AtFunctionThrowsIfIndexOutOfBounds) {
        Vector v(10);
        EXPECT_THROW(float value = v.at(10), Logger::fatal_error);
}

TEST(VectorTest, AdditionOperator) {
        Vector v1     = { 1, 2, 3 };
        Vector v2     = { 4, 5, 6 };
        Vector result = v1 + v2;
        EXPECT_EQ(result, Vector({ 5, 7, 9 }));
}

TEST(VectorTestDebug, AdditionOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(Vector value = v1 + v2, Logger::fatal_error);
}

TEST(VectorTest, AdditionAssignmentOperator) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6 };
        v1       += v2;
        EXPECT_EQ(v1, Vector({ 5, 7, 9 }));
}

TEST(VectorTestDebug, AdditionAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 += v2, Logger::fatal_error);
}

TEST(VectorTest, ScalarAdditionOperator) {
        Vector v      = { 1, 2, 3 };
        Vector result = v + 5;
        EXPECT_EQ(result, Vector({ 6, 7, 8 }));
}

TEST(VectorTest, ScalarAdditionAssignmentOperator) {
        Vector v = { 1, 2, 3 };
        v       += 5;
        EXPECT_EQ(v, Vector({ 6, 7, 8 }));
}

TEST(VectorTest, SubtractionOperator) {
        Vector v1     = { 4, 5, 6 };
        Vector v2     = { 1, 2, 3 };
        Vector result = v1 - v2;
        EXPECT_EQ(result, Vector({ 3, 3, 3 }));
}

TEST(VectorTestDebug, SubtractionOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(Vector value = v1 - v2, Logger::fatal_error);
}

TEST(VectorTest, SubtractionAssignmentOperator) {
        Vector v1 = { 4, 5, 6 };
        Vector v2 = { 1, 2, 3 };
        v1       -= v2;
        EXPECT_EQ(v1, Vector({ 3, 3, 3 }));
}

TEST(VectorTestDebug, SubtractionAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 -= v2, Logger::fatal_error);
}

TEST(VectorTest, ScalarSubtractionOperator) {
        Vector v      = { 4, 5, 6 };
        Vector result = v - 2;
        EXPECT_EQ(result, Vector({ 2, 3, 4 }));
}

TEST(VectorTest, ScalarSubtractionAssignmentOperator) {
        Vector v = { 4, 5, 6 };
        v       -= 2;
        EXPECT_EQ(v, Vector({2, 3, 4}));
}

TEST(VectorTest, MultiplicationOperator) {
        Vector v1     = { 1, 2, 3 };
        Vector v2     = { 2, 3, 4 };
        Vector result = v1 * v2;
        EXPECT_EQ(result, Vector({2, 6, 12}));
}

TEST(VectorTestDebug, MultiplicationOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(Vector value = v1 * v2, Logger::fatal_error);
}

TEST(VectorTest, MultiplicationAssignmentOperator) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 2, 3, 4 };
        v1       *= v2;
        EXPECT_EQ(v1, Vector({ 2, 6, 12 }));
}

TEST(VectorTestDebug, MultiplicationAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 *= v2, Logger::fatal_error);
}

TEST(VectorTest, ScalarMultiplicationOperator) {
        Vector v      = { 1, 2, 3 };
        Vector result = v * 3;
        EXPECT_EQ(result, Vector({ 3, 6, 9 }));
}

TEST(VectorTest, ScalarMultiplicationAssignmentOperator) {
        Vector v = { 1, 2, 3 };
        v       *= 3;
        EXPECT_EQ(v, Vector({ 3, 6, 9 }));
}

TEST(VectorTest, DivisionOperator) {
        Vector v1     = { 4, 9, 16 };
        Vector v2     = { 2, 3, 4 };
        Vector result = v1 / v2;
        EXPECT_EQ(result, Vector({ 2, 3, 4 }));
}

TEST(VectorTestDebug, DivisionOperatorThrowsIfAnyOfTheOtherValuesIs0) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 0 };
        EXPECT_THROW(Vector value = v1 / v2, Logger::fatal_error);
}

TEST(VectorTestDebug, DivisionOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(Vector value = v1 / v2, Logger::fatal_error);
}

TEST(VectorTest, DivisionAssignmentOperator) {
        Vector v1 = { 4, 9, 16 };
        Vector v2 = { 2, 3, 4 };
        v1       /= v2;
        EXPECT_EQ(v1, Vector({ 2, 3, 4 }));
}

TEST(VectorTestDebug, DivisionAssignmentOperatorThrowsIfAnyOfTheOtherValuesIs0) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 0 };
        EXPECT_THROW(v1 /= v2, Logger::fatal_error);
}

TEST(VectorTestDebug, DivisionAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 /= v2, Logger::fatal_error);
}

TEST(VectorTest, ScalarDivisionOperator) {
        Vector v      = { 4, 9, 16 };
        Vector result = v / 2;
        EXPECT_EQ(result, Vector({ 2, 4.5, 8 }));
}

TEST(VectorTestDebug, ScalarDivisionOperatorThrowsIfValueIs0) {
        Vector v1 = { 1, 2, 3 };
        EXPECT_THROW(Vector value = v1 / 0, Logger::fatal_error);
}

TEST(VectorTest, ScalarDivisionAssignmentOperator) {
        Vector v = { 4, 9, 16 };
        v       /= 2;
        EXPECT_EQ(v, Vector({ 2, 4.5, 8 }));
}

TEST(VectorTestDebug, ScalarDivisionAssignmentOperatorThrowsIfValueIs0) {
        Vector v1 = { 1, 2, 3 };
        EXPECT_THROW(v1 /= 0, Logger::fatal_error);
}

TEST(VectorTest, Min) {
        Vector v1     = { 5, 10, 15 };
        Vector v2     = { 7, 8, 12 };
        Vector result = v1.min(v2);
        EXPECT_EQ(result, Vector({ 5, 8, 12 }));
}

TEST(VectorTestDebug, MinThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(Vector value = v1.min(v2), Logger::fatal_error);
}

TEST(VectorTest, ScalarMin) {
        Vector v      = { 5, 10, 15 };
        Vector result = v.min(12);
        EXPECT_EQ(result, Vector({ 5, 10, 12 }));
}

TEST(VectorTest, Max) {
        Vector v1     = { 5, 10, 15 };
        Vector v2     = { 7, 8, 12 };
        Vector result = v1.max(v2);
        EXPECT_EQ(result, Vector({7, 10, 15}));
}

TEST(VectorTestDebug, MaxThrowsIfSizeDoesNotMatch) {
        Vector v1 = { 1, 2, 3 };
        Vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(Vector value = v1.max(v2), Logger::fatal_error);
}

TEST(VectorTest, ScalarMax) {
        Vector v      = { 5, 10, 15 };
        Vector result = v.max(12);
        EXPECT_EQ(result, Vector({ 12, 12, 15 }));
}

TEST(VectorTest, Clamp) {
        Vector v      = { 5, 10, 15 };
        Vector vMin   = { 6, 8, 12 };
        Vector vMax   = { 7, 12, 14 };
        Vector result = v.clamp(vMin, vMax);
        EXPECT_EQ(result, Vector({6, 10, 14}));
}

TEST(VectorTestDebug, ClampThrowsIfMinSizeDoesNotMatch) {
        Vector v    = { 5, 10, 15 };
        Vector vMin = { 6, 8, 12, 11 };
        Vector vMax = { 7, 12, 14 };
        EXPECT_THROW(Vector value = v.clamp(vMin, vMax), Logger::fatal_error);
}

TEST(VectorTestDebug, ClampThrowsIfMaxSizeDoesNotMatch) {
        Vector v    = { 5, 10, 15 };
        Vector vMin = { 6, 8, 12 };
        Vector vMax = { 7, 12, 14, 11 };
        EXPECT_THROW(Vector value = v.clamp(vMin, vMax), Logger::fatal_error);
}

TEST(VectorTest, ScalarClamp) {
        Vector v      = { 5, 10, 15 };
        Vector result = v.clamp(8, 12);
        EXPECT_EQ(result, Vector({8, 10, 12}));
}

