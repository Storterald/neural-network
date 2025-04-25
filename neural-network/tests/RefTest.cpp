#include <gtest/gtest.h>

#include <neural-network/types/Memory.h>
#include <neural-network/Base.h>

TEST(RefTest, ConstructorThrowsWithNullptrOnHost) {
        EXPECT_THROW(Ref<float> r(nullptr, false), Logger::fatal_error);
}

TEST(RefTest, ConstructorThrowsWithNullptrOnDevice) {
        EXPECT_THROW(Ref<float> r(nullptr, true), Logger::fatal_error);
}

TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnHost) {
        float v;
        EXPECT_NO_THROW(Ref<float> r(&v, false));

        Ref<float> r(&v, false);
        EXPECT_EQ(&r, &v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        EXPECT_NO_THROW(Ref<float> r(d_v, true));

        Ref<float> r(d_v, true);
        EXPECT_EQ(&r, d_v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(RefTest, CopyAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        Ref<float> r1(&v1, false);

        float v2 = 12;
        Ref<float> r2(&v2, false);

        EXPECT_NO_THROW(r1 = r2);
        EXPECT_EQ(r1, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r1, &v1);
        EXPECT_NE(&r1, &v2);
}

TEST(RefTest, MoveAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        Ref<float> r1(&v1, false);

        float v2 = 12;

        EXPECT_NO_THROW(r1 = Ref<float>(&v2, false));
        EXPECT_EQ(r1, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r1, &v1);
        EXPECT_NE(&r1, &v2);
}

TEST(RefTest, AssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        Ref<float> r1(&v1, false);

        EXPECT_NO_THROW(r1 = 12);
        EXPECT_EQ(r1, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r1, &v1);
}

TEST(RefTest, ComparisonOperatorReturnsTrueWhenBothReferTheSameValue) {
        float v1 = 13;
        Ref<float> r1(&v1, false);
        Ref<float> r2(&v1, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsTrueWhenReferredValuesAreEqual) {
        float v1 = 13;
        Ref<float> r1(&v1, false);

        float v2 = 13;
        Ref<float> r2(&v2, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsFalseWhenReferredValuesAreNotEqual) {
        float v1 = 13;
        Ref<float> r1(&v1, false);

        float v2 = 12;
        Ref<float> r2(&v2, false);

        EXPECT_FALSE(r1 == r2);
        EXPECT_TRUE(r1 != r2);
}

TEST(RefTest, AddressOperatorReturnsPtrWrapperToTheReferredValue) {
        float v = 13;
        Ref<float> r(&v, false);

        EXPECT_EQ(&r, &v);
}

TEST(RefTest, ImplicitTypeCastOperatorReturnsReferredValue) {
        float v = 13;
        Ref<float> r(&v, false);

        EXPECT_EQ(r, v);
        EXPECT_EQ((float)r, v);
}
