#include <gtest/gtest.h>

#include <concepts>

#include <neural-network/utils/exceptions.h>
#include <neural-network/types/memory.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, TypeAliasesAreCorrectlyInitialized) {
        using ptr = nn::ptr<float>;

        EXPECT_TRUE((std::same_as<ptr::value_type, float>));
        EXPECT_TRUE((std::same_as<ptr::const_value_type, const float>));
        EXPECT_TRUE((std::same_as<ptr::difference_type, ptrdiff_t>));
}

TEST(PtrTest, ConstructorWorksWithNullptrOnHost) {
        EXPECT_NO_THROW(nn::ptr<float> p(nullptr, false));

        const nn::ptr<float> p(nullptr, false);
        EXPECT_EQ(p, nullptr);
        EXPECT_FALSE(p.is_device());
        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ConstructorWorksWithNullptrOnDevice) {
        EXPECT_NO_THROW(nn::ptr<float> p(nullptr, true));

        const nn::ptr<float> p(nullptr, true);
        EXPECT_EQ(p, nullptr);
        EXPECT_TRUE(p.is_device());
        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ConstructorWorksWithValidPointerOnHost) {
        float v = 13;

        EXPECT_NO_THROW(nn::ptr p(&v, false));

        const nn::ptr p(&v, false);
        EXPECT_EQ(p, &v);
        EXPECT_FALSE(p.is_device());
        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ConstructorWorksWithValidPointerOnDevice) {
        float *d_v = nn::cuda::alloc<float>();

        EXPECT_NO_THROW(nn::ptr p(d_v, true));

        const nn::ptr p(d_v, true);
        EXPECT_EQ(p, d_v);
        EXPECT_TRUE(p.is_device());
        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ConstructorSavesValueOnHostWithPointerOnHost) {
        float v = 13;
        const nn::ptr p(&v, false);
        EXPECT_EQ(*p, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ConstructorSavesValueOnHostWithPointerOnDevice) {
        float v = 13;
        float *d_v = nn::cuda::alloc<float>();
        nn::cuda::memcpy(d_v, &v, sizeof(float), cudaMemcpyHostToDevice);

        const nn::ptr p(d_v, true);
        EXPECT_EQ(*p, v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ConstPtrRetainsOriginalType) {
        float v = 13;
        const nn::ptr p(&v, false);
        EXPECT_TRUE((std::same_as<decltype(p)::value_type, float>));
}

TEST(PtrTest, CopyConstructorWorksWithNullptrOnHost) {
        const nn::ptr<float> p(nullptr, false);

        EXPECT_NO_THROW(nn::ptr copy(p));

        const nn::ptr copy(p);
        EXPECT_EQ(copy, nullptr);
        EXPECT_FALSE(copy.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyConstructorWorksWithNullptrOnDevice) {
        const nn::ptr<float> p(nullptr, true);

        EXPECT_NO_THROW(nn::ptr copy(p));

        const nn::ptr copy(p);
        EXPECT_EQ(copy, nullptr);
        EXPECT_TRUE(copy.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyConstructorWorksWithValidPointerOnHost) {
        float v = 13;
        const nn::ptr p(&v, false);

        EXPECT_NO_THROW(nn::ptr copy(p));

        const nn::ptr copy(p);
        EXPECT_EQ(copy, &v);
        EXPECT_FALSE(copy.is_device());
        EXPECT_EQ(*copy, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyConstructorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v = nn::cuda::alloc<float>();
        nn::cuda::memcpy(d_v, &v, sizeof(float), cudaMemcpyHostToDevice);

        const nn::ptr p(d_v, true);

        EXPECT_NO_THROW(nn::ptr copy(p));

        const nn::ptr copy(p);
        EXPECT_EQ(copy, d_v);
        EXPECT_TRUE(copy.is_device());
        EXPECT_EQ(*copy, v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveConstructorWorksWithNullptrOnHost) {
        EXPECT_NO_THROW(nn::ptr p(nn::ptr<float>(nullptr, false)));

        const nn::ptr p(nn::ptr<float>(nullptr, false));
        EXPECT_EQ(p, nullptr);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveConstructorWorksWithNullptrOnDevice) {
        EXPECT_NO_THROW(nn::ptr p(nn::ptr<float>(nullptr, true)));

        const nn::ptr p(nn::ptr<float>(nullptr, true));
        EXPECT_EQ(p, nullptr);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveConstructorWorksWithValidPointerOnHost) {
        float v = 13;
        EXPECT_NO_THROW(nn::ptr p(nn::ptr(&v, false)));

        const nn::ptr p(nn::ptr(&v, false));
        EXPECT_EQ(p, &v);
        EXPECT_FALSE(p.is_device());
        EXPECT_EQ(*p, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveConstructorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v = nn::cuda::alloc<float>();
        nn::cuda::memcpy(d_v, &v, sizeof(float), cudaMemcpyHostToDevice);

        EXPECT_NO_THROW(nn::ptr p(nn::ptr(d_v, true)));

        const nn::ptr p(nn::ptr(d_v, true));
        EXPECT_EQ(p, d_v);
        EXPECT_TRUE(p.is_device());
        EXPECT_EQ(*p, v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyAssignmentOperatorWorksWithNullptrOnHost) {
        const nn::ptr<float> p(nullptr, false);
        nn::ptr<float> copy(nullptr, false);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, nullptr);
        EXPECT_FALSE(copy.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyAssignmentOperatorWorksWithNullptrOnDevice) {
        const nn::ptr<float> p(nullptr, true);
        nn::ptr<float> copy(nullptr, true);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, nullptr);
        EXPECT_TRUE(copy.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, CopyAssignmentOperatorWorksWithValidPointerOnHost) {
        float v = 13;

        const nn::ptr p(&v, false);
        nn::ptr<float> copy(nullptr, false);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, &v);
        EXPECT_FALSE(copy.is_device());
        EXPECT_EQ(*copy, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, CopyAssignmentOperatorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v = nn::cuda::alloc<float>();
        nn::cuda::memcpy(d_v, &v, sizeof(float), cudaMemcpyHostToDevice);

        const nn::ptr p(d_v, true);
        nn::ptr<float> copy(nullptr, true);
        EXPECT_NO_THROW(copy = p);

        EXPECT_EQ(copy, d_v);
        EXPECT_TRUE(copy.is_device());
        EXPECT_EQ(*copy, v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveAssignmentOperatorWorksWithNullptrOnHost) {
        nn::ptr<float> p(nullptr, false);
        EXPECT_NO_THROW(p = nn::ptr<float>(nullptr, false));

        EXPECT_EQ(p, nullptr);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveAssignmentOperatorWorksWithNullptrOnDevice) {
        nn::ptr<float> p(nullptr, true);
        EXPECT_NO_THROW(p = nn::ptr<float>(nullptr, true));

        EXPECT_EQ(p, nullptr);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MoveAssignmentOperatorWorksWithValidPointerOnHost) {
        float v = 13;
        nn::ptr<float> p(nullptr, false);
        EXPECT_NO_THROW(p = nn::ptr(&v, false));

        EXPECT_EQ(p, &v);
        EXPECT_FALSE(p.is_device());
        EXPECT_EQ(*p, v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, MoveAssignmentOperatorWorksWithValidPointerOnDevice) {
        float v = 13;
        float *d_v = nn::cuda::alloc<float>();
        nn::cuda::memcpy(d_v, &v, sizeof(float), cudaMemcpyHostToDevice);

        nn::ptr<float> p(nullptr, true);
        EXPECT_NO_THROW(p = nn::ptr(d_v, true));

        EXPECT_EQ(p, d_v);
        EXPECT_TRUE(p.is_device());
        EXPECT_EQ(*p, v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ComparisonOperatorsWorksWithEqualPointersBothOnHost) {
        float v;
        const nn::ptr a(&v, false);
        const nn::ptr b(&v, false);

        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a != b);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ComparisonOperatorsWorksWithEqualPointersBothOnDevice) {
        float *d_v = nn::cuda::alloc<float>();

        const nn::ptr a(d_v, true);
        const nn::ptr b(d_v, true);

        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a != b);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, ComparisonOperatorsWorksWithDifferentPointersBothOnHost) {
        float v1;
        float v2;
        const nn::ptr a(&v1, false);
        const nn::ptr b(&v2, false);

        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ComparisonOperatorsWorksWithDifferentPointersBothOnDevice) {
        float *d_v1 = nn::cuda::alloc<float>();
        float *d_v2 = nn::cuda::alloc<float>();

        const nn::ptr a(d_v1, true);
        const nn::ptr b(d_v2, true);

        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);

        nn::cuda::free(d_v1);
        nn::cuda::free(d_v2);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, ComparisonOperatorsWorksWithDifferentPointersOnHostAndOnDevice) {
        float v;
        float *d_v = nn::cuda::alloc<float>();

        const nn::ptr a(&v, false);
        const nn::ptr b(d_v, true);

        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, RawComparisonOperatorsWorksWithEqualPointersBothOnHost) {
        float v;
        const nn::ptr a(&v, false);

        EXPECT_TRUE(a == &v);
        EXPECT_FALSE(a != &v);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, RawComparisonOperatorsWorksWithEqualPointersBothOnDevice) {
        float *d_v = nn::cuda::alloc<float>();

        const nn::ptr a(d_v, true);

        EXPECT_TRUE(a == d_v);
        EXPECT_FALSE(a != d_v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, RawComparisonOperatorsWorksWithDifferentPointersBothOnHost) {
        float v1;
        float v2;
        const nn::ptr a(&v1, false);

        EXPECT_FALSE(a == &v2);
        EXPECT_TRUE(a != &v2);
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, RawComparisonOperatorsWorksWithDifferentPointersBothOnDevice) {
        float *d_v1 = nn::cuda::alloc<float>();
        float *d_v2 = nn::cuda::alloc<float>();

        const nn::ptr a(d_v1, true);

        EXPECT_FALSE(a == d_v2);
        EXPECT_TRUE(a != d_v2);

        nn::cuda::free(d_v1);
        nn::cuda::free(d_v2);
}
#endif // BUILD_CUDA_SUPPORT

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, RawComparisonOperatorsWorksWithDifferentPointersOnHostAndOnDevice) {
        float v;
        float *d_v = nn::cuda::alloc<float>();

        const nn::ptr a(&v, false);

        EXPECT_FALSE(a == d_v);
        EXPECT_TRUE(a != d_v);

        nn::cuda::free(d_v);
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, DereferenceOperatorReturnsValidRefWithValidPointer) {
        float v;
        const nn::ptr ptr(&v, false);

        EXPECT_NO_THROW([[maybe_unused]] nn::ref r = *ptr);

        nn::ref r = *ptr;
        EXPECT_EQ(ptr, &r);
}

TEST(PtrTest, DereferenceOperatorThrowsWithNullptr) {
        nn::ptr<float> ptr(nullptr, false);
        EXPECT_THROW([[maybe_unused]] nn::ref r = *ptr, nn::fatal_error);
}

TEST(PtrTest, DereferenceOperatorReturnsRefWithTheSameValueTypeAsThePointer) {
        float v = 0;
        nn::ptr ptr(&v, false);
        const nn::ref r = *ptr;

        EXPECT_TRUE((std::same_as<decltype(r)::value_type, float>));
}

TEST(PtrTest, DereferenceOperatorOnConstPtrReturnsRefWithTheSameValueTypeAsThePointer) {
        float v = 0;
        const nn::ptr ptr(&v, false);
        const nn::ref r = *ptr;

        EXPECT_TRUE((std::same_as<decltype(r)::value_type, float>));
}

TEST(PtrTest, IndexerOperatorReturnsValidRefWithValidPointer) {
        float v;
        const nn::ptr ptr(&v, false);

        EXPECT_NO_THROW([[maybe_unused]] nn::ref r = ptr[0]);

        nn::ref r = ptr[0];
        EXPECT_EQ(ptr, &r);
}

TEST(PtrTest, IndexerOperatorThrowsWithNullptr) {
        const nn::ptr<float> ptr(nullptr, false);
        EXPECT_THROW([[maybe_unused]] nn::ref r = ptr[0], nn::fatal_error);
}

TEST(PtrTest, SumOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        const nn::ptr a(&v[0], false);
        EXPECT_NO_THROW([[maybe_unused]] nn::ptr b = a + 2);

        nn::ptr b = a + 2;
        EXPECT_EQ(b, &v[0] + 2);
        EXPECT_EQ(b, &v[2]);
        EXPECT_EQ(*b, v[2]);
}

TEST(PtrTest, SubOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        const nn::ptr a(&v[2], false);
        EXPECT_NO_THROW([[maybe_unused]] nn::ptr b = a - 2);

        nn::ptr b = a - 2;
        EXPECT_EQ(b, &v[2] - 2);
        EXPECT_EQ(b, &v[0]);
        EXPECT_EQ(*b, v[0]);
}

TEST(PtrTest, SumEqualOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        nn::ptr a(&v[0], false);
        EXPECT_NO_THROW(a += 2);

        EXPECT_EQ(a, &v[0] + 2);
        EXPECT_EQ(a, &v[2]);
        EXPECT_EQ(*a, v[2]);
}

TEST(PtrTest, SubEqualOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        nn::ptr a(&v[2], false);
        EXPECT_NO_THROW(a -= 2);

        EXPECT_EQ(a, &v[2] - 2);
        EXPECT_EQ(a, &v[0]);
        EXPECT_EQ(*a, v[0]);
}

TEST(PtrTest, IncreaseOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        nn::ptr a(&v[0], false);
        EXPECT_NO_THROW([[maybe_unused]] nn::ptr b = a++);

        nn::ptr b = a;
        EXPECT_EQ(b, &v[0] + 1);
        EXPECT_EQ(b, &v[1]);
        EXPECT_EQ(*b, v[1]);
}

TEST(PtrTest, DecreaseOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        nn::ptr a(&v[2], false);
        EXPECT_NO_THROW([[maybe_unused]] nn::ptr b = a--);

        nn::ptr b = a;
        EXPECT_EQ(b, &v[2] - 1);
        EXPECT_EQ(b, &v[1]);
        EXPECT_EQ(*b, v[1]);
}

TEST(PtrTest, IncreaseSelfOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        nn::ptr a(&v[0], false);
        EXPECT_NO_THROW(++a);

        EXPECT_EQ(a, &v[0] + 1);
        EXPECT_EQ(a, &v[1]);
        EXPECT_EQ(*a, v[1]);
}

TEST(PtrTest, DecreaseSelfOperatorReturnsNewValidPointer) {
        float v[3] = { 1, 2, 3 };

        nn::ptr a(&v[2], false);
        EXPECT_NO_THROW(--a);

        EXPECT_EQ(a, &v[2] - 1);
        EXPECT_EQ(a, &v[1]);
        EXPECT_EQ(*a, v[1]);
}

TEST(PtrTest, GetReturnsTheRawPointer) {
        float v;
        const nn::ptr p(&v, false);
        EXPECT_EQ(p.get(), &v);
}

TEST(PtrTest, DeviceReturnsIfThePointerIsADeviceOneWhenOnHost) {
        const nn::ptr<float> p(nullptr, false);
        EXPECT_FALSE(p.is_device());
}

#ifdef BUILD_CUDA_SUPPORT
TEST(PtrTest, DeviceReturnsIfThePointerIsADeviceOneWhenOnDevice) {
        const nn::ptr<float> p(nullptr, true);
        EXPECT_TRUE(p.is_device());
}
#endif // BUILD_CUDA_SUPPORT

TEST(PtrTest, MakeSpanReturnsValidSpan) {
        float v;
        const nn::ptr p(&v, false);
        EXPECT_NO_THROW([[maybe_unused]] nn::span s = p.make_span(1, false));
}
