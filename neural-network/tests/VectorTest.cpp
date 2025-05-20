#include <gtest/gtest.h>

#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>
#include <neural-network/base.h>

TEST(VectorTest, DefaultConstructorInitializesEmptyData) {
        const nn::vector v{};

        EXPECT_EQ(v.view(), nullptr);
        EXPECT_EQ(v.size(), 0);
}

TEST(VectorTest, SizeConstructorAllocatesData) {
        const nn::vector v(10);

        EXPECT_NE(v.view(), nullptr);
        EXPECT_EQ(v.size(), 10);
        EXPECT_NO_FATAL_FAILURE([[maybe_unused]] float value = v[9]);
}

TEST(VectorTest, ConstructorWithFloatArrayWorks) {
        float values[] = { 1.0f, 2.0f, 3.0f };
        const nn::vector v(3, values);

        EXPECT_NE(v.view(), nullptr);
        EXPECT_EQ(v.size(), 3);
        for (uint32_t i = 0; i < 3; ++i)
                EXPECT_EQ(v[i], values[i]);
}

TEST(VectorTest, ConstructorWithInitializerListWorks) {
        auto values = { 1.0f, 2.0f, 3.0f };
        const nn::vector v(values);

        EXPECT_NE(v.view(), nullptr);
        EXPECT_EQ(v.size(), values.size());
        EXPECT_TRUE([&]() -> bool {
                bool value = true;
                for (uint32_t i = 0; i < values.size(); ++i)
                        value &= v[i] == values.begin()[i];

                return value;
        }());
}

TEST(VectorTest, AccessOperator) {
        const nn::vector v = { 1.0f, 2.0f, 3.0f };

        EXPECT_EQ(v[2], 3.0f);
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, AccessOperatorThrowsIfIndexOutOfBounds) {
        const nn::vector v(10);
        EXPECT_THROW([[maybe_unused]] float value = v[10], nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, AtFunction) {
        const nn::vector v = { 1.0f, 2.0f, 3.0f };

        EXPECT_EQ(v.at(2), 3.0f);
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, AtFunctionThrowsIfIndexOutOfBounds) {
        const nn::vector v(10);
        EXPECT_THROW([[maybe_unused]] float value = v.at(10), nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, AdditionOperator) {
        const nn::vector v1     = { 1, 2, 3 };
        const nn::vector v2     = { 4, 5, 6 };
        const nn::vector result = v1 + v2;
        EXPECT_EQ(result, nn::vector({ 5, 7, 9 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, AdditionOperatorThrowsIfSizeDoesNotMatch) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(nn::vector value = v1 + v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, AdditionAssignmentOperator) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6 };
        v1       += v2;
        EXPECT_EQ(v1, nn::vector({ 5, 7, 9 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, AdditionAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 += v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarAdditionOperator) {
        const nn::vector v      = { 1, 2, 3 };
        const nn::vector result = v + 5;
        EXPECT_EQ(result, nn::vector({ 6, 7, 8 }));
}

TEST(VectorTest, ScalarAdditionAssignmentOperator) {
        nn::vector v = { 1, 2, 3 };
        v           += 5;
        EXPECT_EQ(v, nn::vector({ 6, 7, 8 }));
}

TEST(VectorTest, SubtractionOperator) {
        const nn::vector v1     = { 4, 5, 6 };
        const nn::vector v2     = { 1, 2, 3 };
        const nn::vector result = v1 - v2;
        EXPECT_EQ(result, nn::vector({ 3, 3, 3 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, SubtractionOperatorThrowsIfSizeDoesNotMatch) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(nn::vector value = v1 - v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, SubtractionAssignmentOperator) {
        nn::vector v1       = { 4, 5, 6 };
        const nn::vector v2 = { 1, 2, 3 };
        v1                 -= v2;
        EXPECT_EQ(v1, nn::vector({ 3, 3, 3 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, SubtractionAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 -= v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarSubtractionOperator) {
        const nn::vector v      = { 4, 5, 6 };
        const nn::vector result = v - 2;
        EXPECT_EQ(result, nn::vector({ 2, 3, 4 }));
}

TEST(VectorTest, ScalarSubtractionAssignmentOperator) {
        nn::vector v = { 4, 5, 6 };
        v           -= 2;
        EXPECT_EQ(v, nn::vector({2, 3, 4}));
}

TEST(VectorTest, MultiplicationOperator) {
        const nn::vector v1     = { 1, 2, 3 };
        const nn::vector v2     = { 2, 3, 4 };
        const nn::vector result = v1 * v2;
        EXPECT_EQ(result, nn::vector({2, 6, 12}));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, MultiplicationOperatorThrowsIfSizeDoesNotMatch) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(nn::vector value = v1 * v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, MultiplicationAssignmentOperator) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 2, 3, 4 };
        v1                 *= v2;
        EXPECT_EQ(v1, nn::vector({ 2, 6, 12 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, MultiplicationAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 *= v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarMultiplicationOperator) {
        const nn::vector v      = { 1, 2, 3 };
        const nn::vector result = v * 3;
        EXPECT_EQ(result, nn::vector({ 3, 6, 9 }));
}

TEST(VectorTest, ScalarMultiplicationAssignmentOperator) {
        nn::vector v = { 1, 2, 3 };
        v           *= 3;
        EXPECT_EQ(v, nn::vector({ 3, 6, 9 }));
}

TEST(VectorTest, DivisionOperator) {
        const nn::vector v1     = { 4, 9, 16 };
        const nn::vector v2     = { 2, 3, 4 };
        const nn::vector result = v1 / v2;
        EXPECT_EQ(result, nn::vector({ 2, 3, 4 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, DivisionOperatorThrowsIfAnyOfTheOtherValuesIs0) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 0 };
        EXPECT_THROW(nn::vector value = v1 / v2, nn::logger::fatal_error);
}

TEST(VectorTestDebug, DivisionOperatorThrowsIfSizeDoesNotMatch) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(nn::vector value = v1 / v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, DivisionAssignmentOperator) {
        nn::vector v1       = { 4, 9, 16 };
        const nn::vector v2 = { 2, 3, 4 };
        v1                 /= v2;
        EXPECT_EQ(v1, nn::vector({ 2, 3, 4 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, DivisionAssignmentOperatorThrowsIfAnyOfTheOtherValuesIs0) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 0 };
        EXPECT_THROW(v1 /= v2, nn::logger::fatal_error);
}

TEST(VectorTestDebug, DivisionAssignmentOperatorThrowsIfSizeDoesNotMatch) {
        nn::vector v1       = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(v1 /= v2, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarDivisionOperator) {
        const nn::vector v      = { 4, 9, 16 };
        const nn::vector result = v / 2;
        EXPECT_EQ(result, nn::vector({ 2, 4.5, 8 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, ScalarDivisionOperatorThrowsIfValueIs0) {
        const nn::vector v1 = { 1, 2, 3 };
        EXPECT_THROW(nn::vector value = v1 / 0, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarDivisionAssignmentOperator) {
        nn::vector v = { 4, 9, 16 };
        v           /= 2;
        EXPECT_EQ(v, nn::vector({ 2, 4.5, 8 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, ScalarDivisionAssignmentOperatorThrowsIfValueIs0) {
        nn::vector v1 = { 1, 2, 3 };
        EXPECT_THROW(v1 /= 0, nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, Min) {
        const nn::vector v1     = { 5, 10, 15 };
        const nn::vector v2     = { 7, 8, 12 };
        const nn::vector result = v1.min(v2);
        EXPECT_EQ(result, nn::vector({ 5, 8, 12 }));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, MinThrowsIfSizeDoesNotMatch) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(nn::vector value = v1.min(v2), nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarMin) {
        const nn::vector v      = { 5, 10, 15 };
        const nn::vector result = v.min(12);
        EXPECT_EQ(result, nn::vector({ 5, 10, 12 }));
}

TEST(VectorTest, Max) {
        const nn::vector v1     = { 5, 10, 15 };
        const nn::vector v2     = { 7, 8, 12 };
        const nn::vector result = v1.max(v2);
        EXPECT_EQ(result, nn::vector({7, 10, 15}));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, MaxThrowsIfSizeDoesNotMatch) {
        const nn::vector v1 = { 1, 2, 3 };
        const nn::vector v2 = { 4, 5, 6, 7 };
        EXPECT_THROW(nn::vector value = v1.max(v2), nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarMax) {
        const nn::vector v      = { 5, 10, 15 };
        const nn::vector result = v.max(12);
        EXPECT_EQ(result, nn::vector({ 12, 12, 15 }));
}

TEST(VectorTest, Clamp) {
        const nn::vector v      = { 5, 10, 15 };
        const nn::vector vMin   = { 6, 8, 12 };
        const nn::vector vMax   = { 7, 12, 14 };
        const nn::vector result = v.clamp(vMin, vMax);
        EXPECT_EQ(result, nn::vector({6, 10, 14}));
}

#ifdef DEBUG_MODE_ENABLED
TEST(VectorTestDebug, ClampThrowsIfMinSizeDoesNotMatch) {
        const nn::vector v    = { 5, 10, 15 };
        const nn::vector vMin = { 6, 8, 12, 11 };
        const nn::vector vMax = { 7, 12, 14 };
        EXPECT_THROW(nn::vector value = v.clamp(vMin, vMax), nn::logger::fatal_error);
}

TEST(VectorTestDebug, ClampThrowsIfMaxSizeDoesNotMatch) {
        const nn::vector v    = { 5, 10, 15 };
        const nn::vector vMin = { 6, 8, 12 };
        const nn::vector vMax = { 7, 12, 14, 11 };
        EXPECT_THROW(nn::vector value = v.clamp(vMin, vMax), nn::logger::fatal_error);
}
#endif // DEBUG_MODE_ENABLED

TEST(VectorTest, ScalarClamp) {
        const nn::vector v      = { 5, 10, 15 };
        const nn::vector result = v.clamp(8, 12);
        EXPECT_EQ(result, nn::vector({8, 10, 12}));
}
