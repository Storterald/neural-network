#include <gtest/gtest.h>

#include <concepts>

#include <neural-network/types/memory.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#include "CudaTestHelper.h"
#endif // BUILD_CUDA_SUPPORT

TEST(SpanTest, TypeAliasesAreCorrectlyInitialized) {
        using span = nn::span<float>;

        EXPECT_TRUE((std::same_as<span::value_type, float>));
        EXPECT_TRUE((std::same_as<span::const_value_type, const float>));
}

TEST(SpanTest, SpanConstructorDoesNotAllocateMemoryIfLocationsAreHost) {
        float arr[10]{};
        const nn::span s(10, false, arr, false);

        EXPECT_FALSE(s.is_owning());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, SpanConstructorDoesNotAllocateMemoryIfLocationsAreDevice) {
        float *d_arr = nn::cuda::alloc<float>(10);

        const nn::span s(10, true, d_arr, true);

        EXPECT_FALSE(s.is_owning());
        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, SpanConstructorAllocatesMemoryIfSourceIsOnHostAndSpanOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        nn::span s(10, true, arr, false);

        EXPECT_NO_FATAL_FAILURE(helper::access_values(s.size(), s));
        EXPECT_TRUE(helper::check_values(s.size(), s, 1));

        EXPECT_TRUE(s.is_owning());
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, SpanConstructorAllocatesMemoryIfSourceIsOnDeviceAndSpanOnHost) {
        constexpr float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        float *d_arr            = nn::cuda::alloc<float>(10);
        nn::cuda::memcpy(d_arr, arr, 10 * sizeof(float), cudaMemcpyHostToDevice);

        nn::span s(10, false, d_arr, true);

        EXPECT_NO_FATAL_FAILURE([[maybe_unused]] nn::ref<float> v = s[9]);
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s[i], 1);

        EXPECT_TRUE(s.is_owning());
        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, DestructorUpdatesTheDataIfExplicitedInTheConstructor) {
        float *d_arr = nn::cuda::alloc<float>(10);
        nn::cuda::memset(d_arr, 0, 10 * sizeof(float));

        nn::span s(10, false, d_arr, true, true);

        s[3] = 3.0f;

        s.~span();

        float v;
        nn::cuda::memcpy(&v, &d_arr[3], sizeof(float), cudaMemcpyDeviceToHost);

        EXPECT_EQ(v, 3.0f);
        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, ImplicitTypePointerCastReturnsPointerToCorrectLocationWhenCreatedOnHost) {
        float *d_arr = nn::cuda::alloc<float>(10);

        nn::span s(10, false, d_arr, true);

        float *p = s;
        EXPECT_NO_FATAL_FAILURE([[maybe_unused]] float value = *p);
        // While this expectation should be correct, it is not guaranteed to work.
        // EXPECT_DEATH(Helper::access_values(1, p), "");

        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, ImplicitTypePointerCastReturnsPointerToCorrectLocationWhenCreatedOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        nn::span s(10, true, arr, false);

        const float *p = s;
        EXPECT_NO_FATAL_FAILURE(helper::access_values(1, p));
        // While this expectation should be correct, it is not guaranteed to work.
        // EXPECT_DEATH(float value = *p, "");
}
#endif // BUILD_CUDA_SUPPORT

TEST(SpanTest, IndexerReturnsValidReference) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        nn::span s(10, true, arr, false);

        s[3] = 3.0f;
        s.update();

        EXPECT_EQ(arr[3], 3.0f);
}

TEST(SpanTest, IndexerOnConstSpanReturnsReferenceToConst) {
        constexpr float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        const nn::span s(10, true, arr, false);
        const nn::ref r = s[3];

        EXPECT_TRUE((std::same_as<decltype(s)::const_value_type, decltype(r)::value_type>));
}

TEST(SpanTest, SizeReturnsCorrectValue) {
        constexpr float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        const nn::span s(10, true, arr, false);

        EXPECT_EQ(s.size(), 10);
}

TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceAndSpanOnHost) {
        constexpr float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        const nn::span s(10, false, arr, false);

        EXPECT_FALSE(s.is_owning());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceAndSpanOnDevice) {
        float *d_arr = nn::cuda::alloc<float>(10);

        const nn::span s(10, true, d_arr, true);

        EXPECT_FALSE(s.is_owning());
        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceIsOnHostAndSpanOnDevice) {
        constexpr float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        const nn::span s(10, true, arr, false);

        EXPECT_TRUE(s.is_owning());
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceIsOnDeviceAndSpanOnHost) {
        float *d_arr = nn::cuda::alloc<float>(10);
        const nn::span s(10, false, d_arr, true);

        EXPECT_TRUE(s.is_owning());
        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

TEST(SpanTest, UpdateUpdatesSourceCorrectlyWhenSourceOnHostAndSpanOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        nn::span s(10, true, arr, false);

        EXPECT_NE(arr[3], 3.0f);
        s[3] = 3.0f;
        s.update();
        EXPECT_EQ(arr[3], 3.0f);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, UpdateUpdatesSourceCorrectlyWhenSourceOnDeviceAndSpanOnHost) {
        float *d_arr = nn::cuda::alloc<float>(10);
        nn::cuda::memset(d_arr, 0, 10 * sizeof(float));

        nn::span s(10, true, d_arr, false);

        float v;
        nn::cuda::memcpy(&v, &d_arr[3], sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_NE(v, 3.0f);

        s[3] = 3.0f;
        s.update();

        nn::cuda::memcpy(&v, &d_arr[3], sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_EQ(v, 3.0f);
        nn::cuda::free(d_arr);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, UpdateDoesNothingOnSpanOfConstType) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        nn::span<const float> s(10, true, arr, false);

        EXPECT_NE(arr[3], 3.0f);
        helper::set_values(1, (float *)(&s[3]).get(), 3.0f);
        s.update();
        EXPECT_NE(arr[3], 3.0f);
}
#endif // BUILD_CUDA_SUPPORT
