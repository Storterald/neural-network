#include <gtest/gtest.h>

#include <concepts>

#include <neural-network/utils/exceptions.h>
#include <neural-network/types/memory.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#endif // !BUILD_CUDA_SUPPORT

TEST(RefTest, TypeAliasesAreCorrectlyInitialized) {
        using ref = nn::ref<float>;

        EXPECT_TRUE((std::same_as<ref::value_type, float>));
        EXPECT_TRUE((std::same_as<ref::const_value_type, const float>));
}

TEST(RefTest, ConstructorThrowsWithNullptrOnHost) {
        EXPECT_THROW([[maybe_unused]] nn::ref<float> r(nullptr, false), nn::fatal_error);
}

TEST(RefTest, ConstructorThrowsWithNullptrOnDevice) {
        EXPECT_THROW([[maybe_unused]] nn::ref<float> r(nullptr, true), nn::fatal_error);
}

TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnHost) {
        float v;
        EXPECT_NO_THROW([[maybe_unused]] nn::ref r(&v, false));

        const nn::ref r(&v, false);
        EXPECT_EQ(&r, &v);
}

TEST(RefTest, ConstructorDoesNotThrowWithValidPointerOnDevice) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        float *d_v = nn::cuda::alloc<float>();

        EXPECT_NO_THROW([[maybe_unused]] nn::ref r(d_v, true));

        const nn::ref r(d_v, true);
        EXPECT_EQ(&r, d_v);

        nn::cuda::free(d_v);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(RefTest, CopyAssignmentOperatorOnlyChangesValue) {
        float v1 = 13;
        nn::ref r1(&v1, false);

        float v2 = 12;
        const nn::ref r2(&v2, false);

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

TEST(RefTest, AssignmentOperatorCorrectlySetsValueInTheDevice) {
#ifndef BUILD_CUDA_SUPPORT
        GTEST_SKIP() << "Skipping CUDA tests as it's not supported.";
#else // !BUILD_CUDA_SUPPORT
        float *d_v = nn::cuda::alloc<float>();
        nn::ref r(d_v, true);

        EXPECT_NO_THROW(r = 12);
        EXPECT_EQ(r, 12);

        float v;
        nn::cuda::memcpy(&v, d_v, sizeof(float), cudaMemcpyDeviceToHost);

        EXPECT_EQ(v, 12);
        EXPECT_EQ(&r, d_v);

        nn::cuda::free(d_v);
#endif // !BUILD_CUDA_SUPPORT
}

TEST(RefTest, ComparisonOperatorReturnsTrueWhenBothReferTheSameValue) {
        float v1 = 13;
        const nn::ref r1(&v1, false);
        const nn::ref r2(&v1, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsTrueWhenReferredValuesAreEqual) {
        float v1 = 13;
        const nn::ref r1(&v1, false);

        float v2 = 13;
        const nn::ref r2(&v2, false);

        EXPECT_TRUE(r1 == r2);
        EXPECT_FALSE(r1 != r2);
}

TEST(RefTest, ComparisonOperatorReturnsFalseWhenReferredValuesAreNotEqual) {
        float v1 = 13;
        const nn::ref r1(&v1, false);

        float v2 = 12;
        const nn::ref r2(&v2, false);

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
