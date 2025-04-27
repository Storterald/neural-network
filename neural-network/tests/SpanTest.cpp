#include <gtest/gtest.h>

#include <neural-network/types/Memory.h>
#include <neural-network/Base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/CudaBase.h>

#include "CudaTestHelper.h"
#endif // BUILD_CUDA_SUPPORT

USE_NN

TEST(SpanTest, SpanConstructorDoesNotAllocateMemoryIfLocationsAreHost) {
        float arr[10]{};
        Span s(10, false, arr, false);

        EXPECT_FALSE(s.is_owning());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, SpanConstructorDoesNotAllocateMemoryIfLocationsAreDevice) {
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Span s(10, true, d_arr, true);

        EXPECT_FALSE(s.is_owning());
        CUDA_CHECK_ERROR(cudaFree(d_arr), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, SpanConstructorAllocatesMemoryIfSourceIsOnHostAndSpanOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, true, arr, false);

        EXPECT_NO_FATAL_FAILURE(Helper::access_values(s.size(), s));
        EXPECT_TRUE(Helper::check_values(s.size(), s, 1));

        EXPECT_TRUE(s.is_owning());
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, SpanConstructorAllocatesMemoryIfSourceIsOnDeviceAndSpanOnHost) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpy(d_arr, arr, 10 * sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");

        Span s(10, false, d_arr, true);

        EXPECT_NO_FATAL_FAILURE(Ref<float> v = s[9]);
        for (uint32_t i = 0; i < 10; ++i)
                EXPECT_EQ(s[i], 1);

        EXPECT_TRUE(s.is_owning());
        CUDA_CHECK_ERROR(cudaFree(d_arr), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, DestructorUpdatesTheDataIfExplicitedInTheConstructor) {
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemset(d_arr, 0, 10 * sizeof(float)),
                "Failed to set memory in the GPU.");

        Span s(10, false, d_arr, true, true);

        s[3] = 3.0f;

        s.~Span();

        float v;
        CUDA_CHECK_ERROR(cudaMemcpy(&v, &d_arr[3], sizeof(float),
                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");

        EXPECT_EQ(v, 3.0f);

        CUDA_CHECK_ERROR(cudaFree(d_arr), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, ImplicitTypePointerCastReturnsPointerToCorrectLocationWhenCreatedOnHost) {
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Span s(10, false, d_arr, true);

        float *p = s;
        EXPECT_NO_FATAL_FAILURE(float value = *p);
        // While this expectation should be correct, it is not guaranteed to work.
        // EXPECT_DEATH(Helper::access_values(1, p), "");

        CUDA_CHECK_ERROR(cudaFree(d_arr), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, ImplicitTypePointerCastReturnsPointerToCorrectLocationWhenCreatedOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, true, arr, false);

        float *p = s;
        EXPECT_NO_FATAL_FAILURE(Helper::access_values(1, p));
        // While this expectation should be correct, it is not guaranteed to work.
        // EXPECT_DEATH(float value = *p, "");
}
#endif // BUILD_CUDA_SUPPORT

TEST(SpanTest, IndexerReturnsValidReference) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, true, arr, false);

        s[3] = 3.0f;
        s.update();

        EXPECT_EQ(arr[3], 3.0f);
}

TEST(SpanTest, SizeReturnsCorrectValue) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, true, arr, false);

        EXPECT_EQ(s.size(), 10);
}

TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceAndSpanOnHost) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, false, arr, false);

        EXPECT_FALSE(s.is_owning());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceAndSpanOnDevice) {
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Span s(10, true, d_arr, true);

        EXPECT_FALSE(s.is_owning());
        CUDA_CHECK_ERROR(cudaFree(d_arr), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceIsOnHostAndSpanOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, true, arr, false);

        EXPECT_TRUE(s.is_owning());
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, IsOwningReturnsFalseValueWhenOnSourceIsOnDeviceAndSpanOnHost) {
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Span s(10, false, d_arr, true);

        EXPECT_TRUE(s.is_owning());
        CUDA_CHECK_ERROR(cudaFree(d_arr), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(SpanTest, UpdateUpdatesSourceCorrectlyWhenSourceOnHostAndSpanOnDevice) {
        float arr[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        Span s(10, true, arr, false);

        EXPECT_NE(arr[3], 3.0f);
        s[3] = 3.0f;
        s.update();
        EXPECT_EQ(arr[3], 3.0f);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(SpanTest, UpdateUpdatesSourceCorrectlyWhenSourceOnDeviceAndSpanOnHost) {
        float *d_arr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_arr, 10 * sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemset(d_arr, 0, 10 * sizeof(float)),
                "Failed to set memory in the GPU.");

        Span s(10, true, d_arr, false);

        float v;
        CUDA_CHECK_ERROR(cudaMemcpy(&v, &d_arr[3], sizeof(float),
                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");
        EXPECT_NE(v, 3.0f);

        s[3] = 3.0f;
        s.update();

        CUDA_CHECK_ERROR(cudaMemcpy(&v, &d_arr[3], sizeof(float),
                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");

        EXPECT_EQ(v, 3.0f);
}
#endif // BUILD_CUDA_SUPPORT
