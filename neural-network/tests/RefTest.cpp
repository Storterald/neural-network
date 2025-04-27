#include <gtest/gtest.h>

#include <neural-network/types/Memory.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/CudaBase.h>
#endif // BUILD_CUDA_SUPPORT

TEST(RefTest, ConstructorThrowsWithNullptrOnHost) {
        EXPECT_THROW(Ref<float> r(nullptr, false), Logger::fatal_error);
}

TEST(RefTest, ConstructorThrowsWithNullptrOnDevice) {
        EXPECT_THROW(Ref<float> r(nullptr, true), Logger::fatal_error);
}

TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnHost) {
        float v;
        EXPECT_NO_THROW(Ref r(&v, false));

        Ref r(&v, false);
        EXPECT_EQ(&r, &v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        EXPECT_NO_THROW(Ref r(d_v, true));

        Ref r(d_v, true);
        EXPECT_EQ(&r, d_v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(RefTest, CopyAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        Ref r1(&v1, false);

        float v2 = 12;
        Ref r2(&v2, false);

        EXPECT_NO_THROW(r1 = r2);
        EXPECT_EQ(r1, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r1, &v1);
        EXPECT_NE(&r1, &v2);
}

TEST(RefTest, MoveAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        Ref r(&v1, false);

        float v2 = 12;

        EXPECT_NO_THROW(r = Ref(&v2, false));
        EXPECT_EQ(r, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r, &v1);
        EXPECT_NE(&r, &v2);
}

TEST(RefTest, AssignmentOperatorCorrectlySetsValueInTheHost) {
        float v = 13;
        Ref r(&v, false);

        EXPECT_NO_THROW(r = 12);
        EXPECT_EQ(r, 12);
        EXPECT_EQ(v, 12);
        EXPECT_EQ(&r, &v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(RefTest, AssignmentOperatorCorrectlySetsValueInTheDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");
        Ref r(d_v, true);

        EXPECT_NO_THROW(r = 12);
        EXPECT_EQ(r, 12);

        float v;
        CUDA_CHECK_ERROR(cudaMemcpy(&v, d_v, sizeof(float),
                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");

        EXPECT_EQ(v, 12);
        EXPECT_EQ(&r, d_v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(RefTest, ComparisonOperatorReturnsTrueWhenBothReferTheSameValue) {
        float v1 = 13;
        Ref r1(&v1, false);
        Ref r2(&v1, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsTrueWhenReferredValuesAreEqual) {
        float v1 = 13;
        Ref r1(&v1, false);

        float v2 = 13;
        Ref r2(&v2, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsFalseWhenReferredValuesAreNotEqual) {
        float v1 = 13;
        Ref r1(&v1, false);

        float v2 = 12;
        Ref r2(&v2, false);

        EXPECT_FALSE(r1 == r2);
        EXPECT_TRUE(r1 != r2);
}

TEST(RefTest, AddressOperatorReturnsPtrWrapperToTheReferredValue) {
        float v = 13;
        Ref r(&v, false);

        EXPECT_EQ(&r, &v);
}

TEST(RefTest, ImplicitTypeCastOperatorReturnsReferredValue) {
        float v = 13;
        Ref r(&v, false);

        EXPECT_EQ(r, v);
}
