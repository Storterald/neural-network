#include <gtest/gtest.h>

#include <neural-network/types/Memory.h>
#include <neural-network/Base.h>

TEST(PtrTest, ConstructorWorksWithNullptrOnHost) {
        EXPECT_NO_THROW(Ptr<float> p(nullptr, false));

        Ptr<float> p(nullptr, false);
        EXPECT_EQ(p, nullptr);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ConstructorWorksWithNullptrOnDevice) {
        EXPECT_NO_THROW(Ptr<float> p(nullptr, true));

        Ptr<float> p(nullptr, true);
        EXPECT_EQ(p, nullptr);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ConstructorWorksWithValidPointerOnHost) {
        float v = 13;

        EXPECT_NO_THROW(Ptr p(&v, false));

        Ptr p(&v, false);
        EXPECT_EQ(p, &v);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ConstructorWorksWithValidPointerOnDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        EXPECT_NO_THROW(Ptr p(d_v, true));

        Ptr p(d_v, true);
        EXPECT_EQ(p, d_v);
        EXPECT_TRUE(p.is_device());

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ConstructorSavesValueOnHostWithPointerOnHost) {
        float v = 13;
        Ptr p(&v, false);
        EXPECT_EQ(*p, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ConstructorSavesValueOnHostWithPointerOnDevice) {
        float v = 13;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpy(d_v, &v, sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");

        Ptr p(d_v, true);
        EXPECT_EQ(*p, v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyConstructorWorksWithNullptrOnHost) {
        Ptr<float> p(nullptr, false);

        EXPECT_NO_THROW(Ptr copy(p));

        Ptr copy(p);
        EXPECT_EQ(copy, nullptr);
        EXPECT_FALSE(copy.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyConstructorWorksWithNullptrOnDevice) {
        Ptr<float> p(nullptr, true);

        EXPECT_NO_THROW(Ptr copy(p));

        Ptr copy(p);
        EXPECT_EQ(copy, nullptr);
        EXPECT_TRUE(copy.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyConstructorWorksWithValidPointerOnHost) {
        float v = 13;
        Ptr p(&v, false);

        EXPECT_NO_THROW(Ptr copy(p));

        Ptr copy(p);
        EXPECT_EQ(copy, &v);
        EXPECT_FALSE(copy.is_device());
        EXPECT_EQ(*copy, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyConstructorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpy(d_v, &v, sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");

        Ptr p(d_v, true);

        EXPECT_NO_THROW(Ptr copy(p));

        Ptr copy(p);
        EXPECT_EQ(copy, d_v);
        EXPECT_TRUE(copy.is_device());
        EXPECT_EQ(*copy, v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveConstructorWorksWithNullptrOnHost) {
        EXPECT_NO_THROW(Ptr p(Ptr<float>(nullptr, false)));

        Ptr p(Ptr<float>(nullptr, false));
        EXPECT_EQ(p, nullptr);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveConstructorWorksWithNullptrOnDevice) {
        EXPECT_NO_THROW(Ptr p(Ptr<float>(nullptr, true)));

        Ptr p(Ptr<float>(nullptr, true));
        EXPECT_EQ(p, nullptr);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveConstructorWorksWithValidPointerOnHost) {
        float v = 13;
        EXPECT_NO_THROW(Ptr p(Ptr(&v, false)));

        Ptr p(Ptr(&v, false));
        EXPECT_EQ(p, &v);
        EXPECT_FALSE(p.is_device());
        EXPECT_EQ(*p, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveConstructorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpy(d_v, &v, sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");

        EXPECT_NO_THROW(Ptr p(Ptr(d_v, true)));

        Ptr p(Ptr(d_v, true));
        EXPECT_EQ(p, d_v);
        EXPECT_TRUE(p.is_device());
        EXPECT_EQ(*p, v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyAssignmentOperatorWorksWithNullptrOnHost) {
        Ptr<float> p(nullptr, false);
        Ptr<float> copy(nullptr, false);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, nullptr);
        EXPECT_FALSE(copy.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyAssignmentOperatorWorksWithNullptrOnDevice) {
        Ptr<float> p(nullptr, true);
        Ptr<float> copy(nullptr, true);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, nullptr);
        EXPECT_TRUE(copy.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyAssignmentOperatorWorksWithValidPointerOnHost) {
        float v = 13;

        Ptr p(&v, false);
        Ptr<float> copy(nullptr, false);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, &v);
        EXPECT_FALSE(copy.is_device());
        EXPECT_EQ(*copy, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyAssignmentOperatorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpy(d_v, &v, sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");

        Ptr p(d_v, true);
        Ptr<float> copy(nullptr, true);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, d_v);
        EXPECT_TRUE(copy.is_device());
        EXPECT_EQ(*copy, v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveAssignmentOperatorWorksWithNullptrOnHost) {
        Ptr<float> p(nullptr, false);
        EXPECT_NO_THROW(p = Ptr<float>(nullptr, false));

        EXPECT_EQ(p, nullptr);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveAssignmentOperatorWorksWithNullptrOnDevice) {
        Ptr<float> p(nullptr, true);
        EXPECT_NO_THROW(p = Ptr<float>(nullptr, true));

        EXPECT_EQ(p, nullptr);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveAssignmentOperatorWorksWithValidPointerOnHost) {
        float v = 13;
        Ptr<float> p(nullptr, false);
        EXPECT_NO_THROW(p = Ptr(&v, false));

        EXPECT_EQ(p, &v);
        EXPECT_FALSE(p.is_device());
        EXPECT_EQ(*p, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveAssignmentOperatorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpy(d_v, &v, sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");

        Ptr<float> p(nullptr, true);
        EXPECT_NO_THROW(p = Ptr(d_v, true));

        EXPECT_EQ(p, d_v);
        EXPECT_TRUE(p.is_device());
        EXPECT_EQ(*p, v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ComparisonOperatorsWorksWithEqualPointersBothOnHost) {
        float v;
        Ptr a(&v, false);
        Ptr b(&v, false);

        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a != b);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ComparisonOperatorsWorksWithEqualPointersBothOnDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Ptr a(d_v, true);
        Ptr b(d_v, true);

        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a != b);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ComparisonOperatorsWorksWithDifferentPointersBothOnHost) {
        float v1;
        float v2;
        Ptr a(&v1, false);
        Ptr b(&v2, false);

        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ComparisonOperatorsWorksWithDifferentPointersBothOnDevice) {
        float *d_v1;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v1, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        float *d_v2;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v2, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Ptr a(d_v1, true);
        Ptr b(d_v2, true);

        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);

        CUDA_CHECK_ERROR(cudaFree(d_v1), "Failed to free GPU memory.");
        CUDA_CHECK_ERROR(cudaFree(d_v2), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ComparisonOperatorsWorksWithDifferentPointersOnHostAndOnDevice) {
        float v;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Ptr a(&v, false);
        Ptr b(d_v, true);

        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, RawComparisonOperatorsWorksWithEqualPointersBothOnHost) {
        float v;
        Ptr a(&v, false);

        EXPECT_TRUE(a == &v);
        EXPECT_FALSE(a != &v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, RawComparisonOperatorsWorksWithEqualPointersBothOnDevice) {
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Ptr a(d_v, true);

        EXPECT_TRUE(a == d_v);
        EXPECT_FALSE(a != d_v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, RawComparisonOperatorsWorksWithDifferentPointersBothOnHost) {
        float v1;
        float v2;
        Ptr a(&v1, false);

        EXPECT_FALSE(a == &v2);
        EXPECT_TRUE(a != &v2);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, RawComparisonOperatorsWorksWithDifferentPointersBothOnDevice) {
        float *d_v1;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v1, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        float *d_v2;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v2, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Ptr a(d_v1, true);

        EXPECT_FALSE(a == d_v2);
        EXPECT_TRUE(a != d_v2);

        CUDA_CHECK_ERROR(cudaFree(d_v1), "Failed to free GPU memory.");
        CUDA_CHECK_ERROR(cudaFree(d_v2), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, RawComparisonOperatorsWorksWithDifferentPointersOnHostAndOnDevice) {
        float v;
        float *d_v;
        CUDA_CHECK_ERROR(cudaMalloc(&d_v, sizeof(float)),
                "Failed to allocate memory on the GPU.");

        Ptr a(&v, false);

        EXPECT_FALSE(a == d_v);
        EXPECT_TRUE(a != d_v);

        CUDA_CHECK_ERROR(cudaFree(d_v), "Failed to free GPU memory.");
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, DereferenceOperatorReturnsValidRefWithValidPointer) {
        float v;
        Ptr ptr(&v, false);

        EXPECT_NO_THROW(Ref<float> r = *ptr);

        Ref<float> r = *ptr;
        EXPECT_EQ(ptr, &r);
}

TEST(PtrTest, DereferenceOperatorThrowsWithNullptr) {
        Ptr<float> ptr(nullptr, false);
        EXPECT_THROW(Ref<float> r = *ptr, Logger::fatal_error);
}

TEST(PtrTest, IndexerOperatorReturnsValidRefWithValidPointer) {
        float v;
        Ptr ptr(&v, false);

        EXPECT_NO_THROW(Ref<float> r = ptr[0]);

        Ref<float> r = ptr[0];
        EXPECT_EQ(ptr, &r);
}

TEST(PtrTest, IndexerOperatorThrowsWithNullptr) {
        Ptr<float> ptr(nullptr, false);
        EXPECT_THROW(Ref<float> r = ptr[0], Logger::fatal_error);
}

TEST(PtrTest, SumOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[0], false);
        EXPECT_NO_THROW(Ptr<float> b = a + 2);

        Ptr<float> b = a + 2;
        EXPECT_EQ(b, &v[0] + 2);
        EXPECT_EQ(b, &v[2]);
        EXPECT_EQ(*b, v[2]);
}

TEST(PtrTest, SubOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[2], false);
        EXPECT_NO_THROW(Ptr<float> b = a - 2);

        Ptr<float> b = a - 2;
        EXPECT_EQ(b, &v[2] - 2);
        EXPECT_EQ(b, &v[0]);
        EXPECT_EQ(*b, v[0]);
}

TEST(PtrTest, SumEqualOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[0], false);
        EXPECT_NO_THROW(a += 2);

        EXPECT_EQ(a, &v[0] + 2);
        EXPECT_EQ(a, &v[2]);
        EXPECT_EQ(*a, v[2]);
}

TEST(PtrTest, SubEqualOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[2], false);
        EXPECT_NO_THROW(a -= 2);

        EXPECT_EQ(a, &v[2] - 2);
        EXPECT_EQ(a, &v[0]);
        EXPECT_EQ(*a, v[0]);
}

TEST(PtrTest, IncreaseOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[0], false);
        EXPECT_NO_THROW(Ptr<float> b = a++);

        Ptr<float> b = a++;
        EXPECT_EQ(b, &v[0] + 1);
        EXPECT_EQ(b, &v[1]);
        EXPECT_EQ(*b, v[1]);
}

TEST(PtrTest, DecreaseOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[2], false);
        EXPECT_NO_THROW(Ptr<float> b = a--);

        Ptr<float> b = a--;
        EXPECT_EQ(b, &v[2] - 1);
        EXPECT_EQ(b, &v[1]);
        EXPECT_EQ(*b, v[1]);
}

TEST(PtrTest, IncreaseSelfOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[0], false);
        EXPECT_NO_THROW(++a);

        EXPECT_EQ(a, &v[0] + 1);
        EXPECT_EQ(a, &v[1]);
        EXPECT_EQ(*a, v[1]);
}

TEST(PtrTest, DecreaseSelfOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        Ptr a(&v[2], false);
        EXPECT_NO_THROW(--a);

        EXPECT_EQ(a, &v[2] - 1);
        EXPECT_EQ(a, &v[1]);
        EXPECT_EQ(*a, v[1]);
}

TEST(PtrTest, GetReturnsTheRawPointer) {
        float v;
        Ptr p(&v, false);
        EXPECT_EQ(p.get(), &v);
}

TEST(PtrTest, DeviceReturnsIfThePointerIsADeviceOneWhenOnHost) {
        Ptr<float> p(nullptr, false);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, DeviceReturnsIfThePointerIsADeviceOneWhenOnDevice) {
        Ptr<float> p(nullptr, true);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, SpanReturnsValidSpan) {
        float v;
        Ptr p(&v, false);
        EXPECT_NO_THROW(Span<float> s = p.span(1, false));
}
