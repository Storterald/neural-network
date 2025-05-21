#pragma once

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <immintrin.h> // AVX, AVX512

#include <concepts>
#include <cstdint>

#ifdef _mm256_cmp_ps_mask
#undef _mm256_cmp_ps_mask
#endif // _mm256_cmp_ps_mask

#define _mm256_cmp_ps_mask(a, b, cmp) ((__mmask8)_mm256_movemask_ps(_mm256_cmp_ps(a, b, cmp)))

#define _mm_cmp_ps__CMP_EQ_OQ  _mm_cmpeq_ps
#define _mm_cmp_ps__CMP_NEQ_OQ _mm_cmpneq_ps
#define _mm_cmp_ps__CMP_GE_OQ  _mm_cmpge_ps
#define _mm_cmp_ps__CMP_GT_OQ  _mm_cmpgt_ps
#define _mm_cmp_ps__CMP_LE_OQ  _mm_cmple_ps
#define _mm_cmp_ps__CMP_LT_OQ  _mm_cmplt_ps

#define _mm_cmp_ps_mask(a, b, cmp) _mm_movemask_ps(_mm_cmp_ps_##cmp(a, b))

namespace nn::simd {

class mask4;
class mask8;
class mask16;
class m128;
class m256;
class m512;

class mask16 {
public:
        static constexpr __mmask16 ones = 0xFFFF;

        inline mask16(__mmask16 mask) : data(mask) {}
        inline operator __mmask16() const { return data; }

        [[nodiscard]] inline m512 blend(const m512 &first, const m512 &second) const;

private:
        __mmask16 data;

}; // mask16

class mask8 {
public:
        static constexpr __mmask8 ones = 0xFF;

        inline mask8(__mmask8 mask) : data(mask) {}
        inline operator __mmask8() const { return data; }

        [[nodiscard]] inline m256 blend(const m256 &first, const m256 &second) const;

private:
        __mmask8 data;

}; // mask8

class mask4 {
public:
        static constexpr int ones = 0x0F;

        inline mask4(int mask) : data(mask) {}
        inline operator int() const { return data; }

        [[nodiscard]] inline m128 blend(const m128 &first, const m128 &second) const;

private:
        int data;

}; // mask4

class m512 {
public:
        static constexpr uint32_t width = 16;
        using mask = mask16;

        inline m512() : data(_mm512_setzero_ps()) {}
        inline m512(const float *values) : data(_mm512_load_ps(values)) {}
        inline m512(float scalar) : data(_mm512_set1_ps(scalar)) {}
        inline m512(const __m512 &reg) : data(reg) {}
        inline operator __m512() const { return data; }

        inline void store(float *dst) const;

        [[nodiscard]] inline m512 operator& (const m512 &other) const;
        [[nodiscard]] inline m512 operator| (const m512 &other) const;
        [[nodiscard]] inline m512 operator^ (const m512 &other) const;

        [[nodiscard]] inline m512 operator+ (const m512 &other) const;
        [[nodiscard]] inline m512 operator- (const m512 &other) const;
        [[nodiscard]] inline m512 operator* (const m512 &other) const;
        [[nodiscard]] inline m512 operator/ (const m512 &other) const;

        inline m512 &operator+= (const m512 &other);
        inline m512 &operator-= (const m512 &other);
        inline m512 &operator*= (const m512 &other);
        inline m512 &operator/= (const m512 &other);

        [[nodiscard]] inline mask operator> (const m512 &other) const;
        [[nodiscard]] inline mask operator< (const m512 &other) const;
        [[nodiscard]] inline mask operator>= (const m512 &other) const;
        [[nodiscard]] inline mask operator<= (const m512 &other) const;
        [[nodiscard]] inline mask operator== (const m512 &other) const;
        [[nodiscard]] inline mask operator!= (const m512 &other) const;

        [[nodiscard]] inline m512 abs() const;
        [[nodiscard]] inline m512 fma(const m512 &b, const m512 &c) const;
        [[nodiscard]] inline m512 fnma(const m512 &b, const m512 &c) const;
        [[nodiscard]] inline m512 max(const m512 &other) const;
        [[nodiscard]] inline m512 min(const m512 &other) const;

        [[nodiscard]] inline float sum() const;

private:
        __m512 data;

}; // class m512

class m256 {
public:
        static constexpr uint32_t width = 8;
        using mask = mask8;

        inline m256() : data(_mm256_setzero_ps()) {}
        inline m256(const float *values) : data(_mm256_load_ps(values)) {}
        inline m256(float scalar) : data(_mm256_set1_ps(scalar)) {}
        inline m256(const __m256 &reg) : data(reg) {}
        inline operator __m256() const { return data; }

        inline void store(float *dst) const;

        [[nodiscard]] inline m256 operator& (const m256 &other) const;
        [[nodiscard]] inline m256 operator| (const m256 &other) const;
        [[nodiscard]] inline m256 operator^ (const m256 &other) const;

        [[nodiscard]] inline m256 operator+ (const m256 &other) const;
        [[nodiscard]] inline m256 operator- (const m256 &other) const;
        [[nodiscard]] inline m256 operator* (const m256 &other) const;
        [[nodiscard]] inline m256 operator/ (const m256 &other) const;

        inline m256 &operator+= (const m256 &other);
        inline m256 &operator-= (const m256 &other);
        inline m256 &operator*= (const m256 &other);
        inline m256 &operator/= (const m256 &other);

        [[nodiscard]] inline mask operator> (const m256 &other) const;
        [[nodiscard]] inline mask operator< (const m256 &other) const;
        [[nodiscard]] inline mask operator>= (const m256 &other) const;
        [[nodiscard]] inline mask operator<= (const m256 &other) const;
        [[nodiscard]] inline mask operator== (const m256 &other) const;
        [[nodiscard]] inline mask operator!= (const m256 &other) const;

        [[nodiscard]] inline m256 abs() const;
        [[nodiscard]] inline m256 fma(const m256 &b, const m256 &c) const;
        [[nodiscard]] inline m256 fnma(const m256 &b, const m256 &c) const;
        [[nodiscard]] inline m256 max(const m256 &other) const;
        [[nodiscard]] inline m256 min(const m256 &other) const;

        [[nodiscard]] inline float sum() const;

private:
        __m256 data;

}; // class m256

class m128 {
public:
        static constexpr uint32_t width = 4;
        using mask = mask4;

        inline m128() : data(_mm_setzero_ps()) {}
        inline m128(const float *values) : data(_mm_load_ps(values)) {}
        inline m128(float scalar) : data(_mm_set1_ps(scalar)) {}
        inline m128(const __m128 &reg) : data(reg) {}
        inline operator __m128() const { return data; }

        inline void store(float *dst) const;

        [[nodiscard]] inline m128 operator& (const m128 &other) const;
        [[nodiscard]] inline m128 operator| (const m128 &other) const;
        [[nodiscard]] inline m128 operator^ (const m128 &other) const;

        [[nodiscard]] inline m128 operator+ (const m128 &other) const;
        [[nodiscard]] inline m128 operator- (const m128 &other) const;
        [[nodiscard]] inline m128 operator* (const m128 &other) const;
        [[nodiscard]] inline m128 operator/ (const m128 &other) const;

        inline m128 &operator+= (const m128 &other);
        inline m128 &operator-= (const m128 &other);
        inline m128 &operator*= (const m128 &other);
        inline m128 &operator/= (const m128 &other);

        [[nodiscard]] inline mask operator> (const m128 &other) const;
        [[nodiscard]] inline mask operator< (const m128 &other) const;
        [[nodiscard]] inline mask operator>= (const m128 &other) const;
        [[nodiscard]] inline mask operator<= (const m128 &other) const;
        [[nodiscard]] inline mask operator== (const m128 &other) const;
        [[nodiscard]] inline mask operator!= (const m128 &other) const;

        [[nodiscard]] inline m128 abs() const;
        [[nodiscard]] inline m128 fma(const m128 &b, const m128 &c) const;
        [[nodiscard]] inline m128 fnma(const m128 &b, const m128 &c) const;
        [[nodiscard]] inline m128 max(const m128 &other) const;
        [[nodiscard]] inline m128 min(const m128 &other) const;

        [[nodiscard]] inline float sum() const;

private:
        __m128 data;

}; // class m128

inline void m512::store(float *dst) const
{
        _mm512_storeu_ps(dst, data);
}

inline m512 m512::operator& (const m512 &other) const
{
        return _mm512_and_ps(data, other);
}

inline m512 m512::operator| (const m512 &other) const
{
        return _mm512_or_ps(data, other);
}

inline m512 m512::operator^ (const m512 &other) const
{
        return _mm512_xor_ps(data, other);
}

inline m512 m512::operator+ (const m512 &other) const
{
        return _mm512_add_ps(data, other);
}

inline m512 m512::operator- (const m512 &other) const
{
        return _mm512_sub_ps(data, other);
}

inline m512 m512::operator* (const m512 &other) const
{
        return _mm512_mul_ps(data, other);
}

inline m512 m512::operator/ (const m512 &other) const
{
        return _mm512_div_ps(data, other);
}

inline m512 &m512::operator+= (const m512 &other)
{
        data = _mm512_add_ps(data, other);
        return *this;
}

inline m512 &m512::operator-= (const m512 &other)
{
        data = _mm512_sub_ps(data, other);
        return *this;
}

inline m512 &m512::operator*= (const m512 &other)
{
        data = _mm512_mul_ps(data, other);
        return *this;
}

inline m512 &m512::operator/= (const m512 &other)
{
        data = _mm512_div_ps(data, other);
        return *this;
}

inline m512::mask m512::operator> (const m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_GT_OQ);
}

inline m512::mask m512::operator< (const m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_LT_OQ);
}

inline m512::mask m512::operator>= (const m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_GE_OQ);
}

inline m512::mask m512::operator<= (const m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_LE_OQ);
}

inline m512::mask m512::operator== (const m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_EQ_OQ);
}

inline m512::mask m512::operator!= (const m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_NEQ_OQ);
}

inline m512 m512::abs() const
{
        return _mm512_abs_ps(data);
}

inline m512 m512::fma(const m512 &b, const m512 &c) const
{
        return _mm512_fmadd_ps(data, b, c);
}

inline m512 m512::fnma(const m512 &b, const m512 &c) const
{
        return _mm512_fnmadd_ps(data, b, c);
}

inline m512 m512::max(const m512 &other) const
{
        return _mm512_max_ps(data, other);
}

inline m512 m512::min(const m512 &other) const
{
        return _mm512_min_ps(data, other);
}

inline float m512::sum() const
{
        return _mm512_reduce_add_ps(data);
}

inline void m256::store(float *dst) const
{
        _mm256_storeu_ps(dst, data);
}

inline m256 m256::operator& (const m256 &other) const
{
        return _mm256_and_ps(data, other);
}

inline m256 m256::operator| (const m256 &other) const
{
        return _mm256_or_ps(data, other);
}

inline m256 m256::operator^ (const m256 &other) const
{
        return _mm256_xor_ps(data, other);
}

inline m256 m256::operator+ (const m256 &other) const
{
        return _mm256_add_ps(data, other);
}

inline m256 m256::operator- (const m256 &other) const
{
        return _mm256_sub_ps(data, other);
}

inline m256 m256::operator* (const m256 &other) const
{
        return _mm256_mul_ps(data, other);
}

inline m256 m256::operator/ (const m256 &other) const
{
        return _mm256_div_ps(data, other);
}

inline m256 &m256::operator+= (const m256 &other)
{
        data = _mm256_add_ps(data, other);
        return *this;
}

inline m256 &m256::operator-= (const m256 &other)
{
        data = _mm256_sub_ps(data, other);
        return *this;
}

inline m256 &m256::operator*= (const m256 &other)
{
        data = _mm256_mul_ps(data, other);
        return *this;
}

inline m256 &m256::operator/= (const m256 &other)
{
        data = _mm256_div_ps(data, other);
        return *this;
}

inline m256::mask m256::operator> (const m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_GT_OQ);
}

inline m256::mask m256::operator< (const m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_LT_OQ);
}

inline m256::mask m256::operator>= (const m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_GE_OQ);
}

inline m256::mask m256::operator<= (const m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_LE_OQ);
}

inline m256::mask m256::operator== (const m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_EQ_OQ);
}

inline m256::mask m256::operator!= (const m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_NEQ_OQ);
}

inline m256 m256::abs() const
{
        return _mm256_and_ps(data, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}

inline m256 m256::fma(const m256 &b, const m256 &c) const
{
        return _mm256_fmadd_ps(data, b, c);
}

inline m256 m256::fnma(const m256 &b, const m256 &c) const
{
        return _mm256_fnmadd_ps(data, b, c);
}

inline m256 m256::max(const m256 &other) const
{
        return _mm256_max_ps(data, other);
}

inline m256 m256::min(const m256 &other) const
{
        return _mm256_min_ps(data, other);
}

inline float m256::sum() const
{
        const __m128 low    = _mm256_castps256_ps128(data);
        const __m128 high   = _mm256_extractf128_ps(data, 1);
        const __m128 sum128 = _mm_add_ps(low, high);
        __m128 shuf         = _mm_movehdup_ps(sum128);
        __m128 sums         = _mm_add_ps(sum128, shuf);
        shuf                = _mm_movehl_ps(shuf, sums);
        sums                = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

inline void m128::store(float *dst) const
{
        _mm_storeu_ps(dst, data);
}

inline m128 m128::operator& (const m128 &other) const
{
        return _mm_and_ps(data, other);
}

inline m128 m128::operator| (const m128 &other) const
{
        return _mm_or_ps(data, other);
}

inline m128 m128::operator^ (const m128 &other) const
{
        return _mm_xor_ps(data, other);
}

inline m128 m128::operator+ (const m128 &other) const
{
        return _mm_add_ps(data, other);
}

inline m128 m128::operator- (const m128 &other) const
{
        return _mm_sub_ps(data, other);
}

inline m128 m128::operator* (const m128 &other) const
{
        return _mm_mul_ps(data, other);
}

inline m128 m128::operator/ (const m128 &other) const
{
        return _mm_div_ps(data, other);
}

inline m128 &m128::operator+= (const m128 &other)
{
        data = _mm_add_ps(data, other);
        return *this;
}

inline m128 &m128::operator-= (const m128 &other)
{
        data = _mm_sub_ps(data, other);
        return *this;
}

inline m128 &m128::operator*= (const m128 &other)
{
        data = _mm_mul_ps(data, other);
        return *this;
}

inline m128 &m128::operator/= (const m128 &other)
{
        data = _mm_div_ps(data, other);
        return *this;
}

inline m128::mask m128::operator> (const m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_GT_OQ);
}

inline m128::mask m128::operator< (const m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_LT_OQ);
}

inline m128::mask m128::operator>= (const m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_GE_OQ);
}

inline m128::mask m128::operator<= (const m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_LE_OQ);
}

inline m128::mask m128::operator== (const m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_EQ_OQ);
}

inline m128::mask m128::operator!= (const m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_NEQ_OQ);
}

inline m128 m128::abs() const
{
        return _mm_and_ps(data, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}

inline m128 m128::fma(const m128 &b, const m128 &c) const
{
        return _mm_add_ps(_mm_mul_ps(data, b), c);
}

inline m128 m128::fnma(const m128 &b, const m128 &c) const
{
        return _mm_sub_ps(c, _mm_mul_ps(data, b));
}

inline m128 m128::max(const m128 &other) const
{
        return _mm_max_ps(data, other);
}

inline m128 m128::min(const m128 &other) const
{
        return _mm_min_ps(data, other);
}

inline float m128::sum() const
{
        __m128 shuf = _mm_movehdup_ps(data);
        __m128 sums = _mm_add_ps(data, shuf);
        shuf        = _mm_movehl_ps(shuf, sums);
        sums        = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

inline m512 mask16::blend(const m512 &first, const m512 &second) const
{
        return _mm512_mask_blend_ps(data, first, second);
}

inline m256 mask8::blend(const m256 &first, const m256 &second) const
{
        const __m256i blend_mask = _mm256_set_epi32(
                (data & 0x80) * -1,
                (data & 0x40) * -1,
                (data & 0x20) * -1,
                (data & 0x10) * -1,
                (data & 0x08) * -1,
                (data & 0x04) * -1,
                (data & 0x02) * -1,
                (data & 0x01) * -1);

        const __m256 ps = _mm256_castsi256_ps(blend_mask);

        return _mm256_or_ps(
                _mm256_andnot_ps(ps, first),
                _mm256_and_ps(ps, second)
        );
}

inline m128 mask4::blend(const m128 &first, const m128 &second) const
{
        const __m128i blend_mask = _mm_set_epi32(
               (data & 0x8) * -1,
               (data & 0x4) * -1,
               (data & 0x2) * -1,
               (data & 0x1) * -1);

        const __m128 ps = _mm_castsi128_ps(blend_mask);

        return _mm_or_ps(
                _mm_andnot_ps(ps, first),
                _mm_and_ps(ps, second)
        );
}

template<typename T>
concept simd = requires(T v)
{
        T::width;
        typename T::mask;
        T::mask::ones;

        requires std::default_initializable<T>;
        requires std::constructible_from<T, float>;
        requires std::constructible_from<T, float *>;

        { v & v } -> std::same_as<T>;
        { v | v } -> std::same_as<T>;
        { v ^ v } -> std::same_as<T>;

        { v + v } -> std::same_as<T>;
        { v - v } -> std::same_as<T>;
        { v * v } -> std::same_as<T>;
        { v / v } -> std::same_as<T>;

        { v += v } -> std::same_as<T &>;
        { v -= v } -> std::same_as<T &>;
        { v *= v } -> std::same_as<T &>;
        { v /= v } -> std::same_as<T &>;

        { v > v  } -> std::same_as<typename T::mask>;
        { v < v  } -> std::same_as<typename T::mask>;
        { v >= v } -> std::same_as<typename T::mask>;
        { v <= v } -> std::same_as<typename T::mask>;
        { v == v } -> std::same_as<typename T::mask>;
        { v != v } -> std::same_as<typename T::mask>;

        { v.abs() } -> std::same_as<T>;
        { v.fma(v, v) } -> std::same_as<T>;
        { v.fnma(v, v) } -> std::same_as<T>;
        { v.max(v) } -> std::same_as<T>;
        { v.min(v) } -> std::same_as<T>;
        { v.sum() } -> std::same_as<float>;
};

} // namespace nn::simd

static_assert(nn::simd::simd<nn::simd::m512>);
static_assert(nn::simd::simd<nn::simd::m256>);
static_assert(nn::simd::simd<nn::simd::m128>);
