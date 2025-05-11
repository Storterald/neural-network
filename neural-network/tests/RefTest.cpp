#include <gtest/gtest.h>

#include <concepts>

#include <neural-network/types/memory.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

TEST(RefTest, TypeAliasesAreCorrectlyInitialized) {
        using ref = nn::ref<float>;

        EXPECT_TRUE((std::same_as<ref::value_type, float>));
        EXPECT_TRUE((std::same_as<ref::const_value_type, const float>));
}

TEST(RefTest, ConstructorThrowsWithNullptrOnHost) {
        EXPECT_THROW(nn::ref<float> r(nullptr, false), nn::logger::fatal_error);
}

TEST(RefTest, ConstructorThrowsWithNullptrOnDevice) {
        EXPECT_THROW(nn::ref<float> r(nullptr, true), nn::logger::fatal_error);
}

TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnHost) {
        float v;
        EXPECT_NO_THROW(nn::ref r(&v, false));

        nn::ref r(&v, false);
        EXPECT_EQ(&r, &v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        EXPECT_NO_THROW(nn::ref r(d_v, true));

        nn::ref r(d_v, true);
        EXPECT_EQ(&r, d_v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(RefTest, CopyAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        nn::ref r1(&v1, false);

        float v2 = 12;
        nn::ref r2(&v2, false);

        EXPECT_NO_THROW(r1 = r2);
        EXPECT_EQ(r1, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r1, &v1);
        EXPECT_NE(&r1, &v2);
}

TEST(RefTest, MoveAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        nn::ref r(&v1, false);

        float v2 = 12;

        EXPECT_NO_THROW(r = nn::ref(&v2, false));
        EXPECT_EQ(r, 12);
        EXPECT_EQ(v1, 12);
        EXPECT_EQ(&r, &v1);
        EXPECT_NE(&r, &v2);
}

TEST(RefTest, AssignmentOperatorCorrectlySetsValueInTheHost) {
        float v = 13;
        nn::ref r(&v, false);

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
        nn::ref r(d_v, true);

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
        nn::ref r1(&v1, false);
        nn::ref r2(&v1, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsTrueWhenReferredValuesAreEqual) {
        float v1 = 13;
        nn::ref r1(&v1, false);

        float v2 = 13;
        nn::ref r2(&v2, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsFalseWhenReferredValuesAreNotEqual) {
        float v1 = 13;
        nn::ref r1(&v1, false);

        float v2 = 12;
        nn::ref r2(&v2, false);

        EXPECT_FALSE(r1 == r2);
        EXPECT_TRUE(r1 != r2);
}

TEST(RefTest, AddressOperatorReturnsPtrWrapperToTheReferredValue) {
        float v = 13;
        const nn::ref r(&v, false);
        const nn::ptr p = &r;

        EXPECT_EQ(p, &v);
}

TEST(RefTest, AddressOperatorReturnsPtrWrapperWithTheSameValueTypeAsTheRef) {
        float v = 13;
        nn::ref r(&v, false);
        const nn::ptr p = &r;

        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));
}

TEST(RefTest, AddressOperatorOnConstRefReturnsPtrWithTheSameValueTypeAsTheRef) {
        float v = 13;
        const nn::ref r(&v, false);
        const nn::ptr p = &r;

        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));
}

TEST(RefTest, ImplicitTypeCastOperatorReturnsReferredValue) {
        float v = 13;
        const nn::ref r(&v, false);
        EXPECT_EQ(r, v);
}
