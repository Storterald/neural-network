#pragma once

#ifndef TARGET_X86_64
#error utils/_simd256.h cannot be included if the architecture is not x86_64
#endif // !TARGET_X86_64

#if SIMD_SUPPORT_LEVEL < 2
#error utils/_simd256.h cannot be included if the CPU does not support AVX
#endif // SIMD_SUPPORT_LEVEL < 2

#include <immintrin.h>

#include <cstdint>

#ifdef _mm256_cmp_ps_mask
#undef _mm256_cmp_ps_mask
#endif // _mm256_cmp_ps_mask

#define _mm256_cmp_ps_mask(a, b, cmp) ((__mmask8)_mm256_movemask_ps(_mm256_cmp_ps(a, b, cmp)))

namespace nn::simd {

class _m256;
class _mask8 {
public:
        static constexpr __mmask8 ones = 0xFF;

        inline _mask8(__mmask8 mask) : data(mask) {}
        inline operator __mmask8() const { return data; }

        [[nodiscard]] inline _m256 operator() (const _m256 &first, const _m256 &second) const;

private:
        __mmask8 data;

}; // _mask8

class _m256 {
public:
        static constexpr uint32_t width     = 8;
        using mask                          = _mask8;

        inline _m256() : data(_mm256_setzero_ps()) {}
        inline _m256(const float *values) : data(_mm256_load_ps(values)) {}
        inline _m256(float scalar) : data(_mm256_set1_ps(scalar)) {}
        inline _m256(const __m256 &reg) : data(reg) {}
        inline operator __m256() const { return data; }

        inline void store(float *dst) const;

        [[nodiscard]] inline _m256 operator& (const _m256 &other) const;
        [[nodiscard]] inline _m256 operator| (const _m256 &other) const;
        [[nodiscard]] inline _m256 operator^ (const _m256 &other) const;

        [[nodiscard]] inline _m256 operator+ (const _m256 &other) const;
        [[nodiscard]] inline _m256 operator- (const _m256 &other) const;
        [[nodiscard]] inline _m256 operator* (const _m256 &other) const;
        [[nodiscard]] inline _m256 operator/ (const _m256 &other) const;

        inline _m256 &operator+= (const _m256 &other);
        inline _m256 &operator-= (const _m256 &other);
        inline _m256 &operator*= (const _m256 &other);
        inline _m256 &operator/= (const _m256 &other);

        [[nodiscard]] inline mask operator> (const _m256 &other) const;
        [[nodiscard]] inline mask operator< (const _m256 &other) const;
        [[nodiscard]] inline mask operator>= (const _m256 &other) const;
        [[nodiscard]] inline mask operator<= (const _m256 &other) const;
        [[nodiscard]] inline mask operator== (const _m256 &other) const;
        [[nodiscard]] inline mask operator!= (const _m256 &other) const;

        [[nodiscard]] inline _m256 abs() const;
        [[nodiscard]] inline _m256 fma(const _m256 &b, const _m256 &c) const;
        [[nodiscard]] inline _m256 fnma(const _m256 &b, const _m256 &c) const;
        [[nodiscard]] inline _m256 max(const _m256 &other) const;
        [[nodiscard]] inline _m256 min(const _m256 &other) const;

        [[nodiscard]] inline float sum() const;

private:
        __m256 data;

}; // class _m256

inline void _m256::store(float *dst) const
{
        _mm256_storeu_ps(dst, data);
}

inline _m256 _m256::operator& (const _m256 &other) const
{
        return _mm256_and_ps(data, other);
}

inline _m256 _m256::operator| (const _m256 &other) const
{
        return _mm256_or_ps(data, other);
}

inline _m256 _m256::operator^ (const _m256 &other) const
{
        return _mm256_xor_ps(data, other);
}

inline _m256 _m256::operator+ (const _m256 &other) const
{
        return _mm256_add_ps(data, other);
}

inline _m256 _m256::operator- (const _m256 &other) const
{
        return _mm256_sub_ps(data, other);
}

inline _m256 _m256::operator* (const _m256 &other) const
{
        return _mm256_mul_ps(data, other);
}

inline _m256 _m256::operator/ (const _m256 &other) const
{
        return _mm256_div_ps(data, other);
}

inline _m256 &_m256::operator+= (const _m256 &other)
{
        data = _mm256_add_ps(data, other);
        return *this;
}

inline _m256 &_m256::operator-= (const _m256 &other)
{
        data = _mm256_sub_ps(data, other);
        return *this;
}

inline _m256 &_m256::operator*= (const _m256 &other)
{
        data = _mm256_mul_ps(data, other);
        return *this;
}

inline _m256 &_m256::operator/= (const _m256 &other)
{
        data = _mm256_div_ps(data, other);
        return *this;
}

inline _m256::mask _m256::operator> (const _m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_GT_OQ);
}

inline _m256::mask _m256::operator< (const _m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_LT_OQ);
}

inline _m256::mask _m256::operator>= (const _m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_GE_OQ);
}

inline _m256::mask _m256::operator<= (const _m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_LE_OQ);
}

inline _m256::mask _m256::operator== (const _m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_EQ_OQ);
}

inline _m256::mask _m256::operator!= (const _m256 &other) const
{
        return _mm256_cmp_ps_mask(data, other, _CMP_NEQ_OQ);
}

inline _m256 _m256::abs() const
{
        return _mm256_and_ps(data, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}

inline _m256 _m256::fma(const _m256 &b, const _m256 &c) const
{
        return _mm256_fmadd_ps(data, b, c);
}

inline _m256 _m256::fnma(const _m256 &b, const _m256 &c) const
{
        return _mm256_fnmadd_ps(data, b, c);
}

inline _m256 _m256::max(const _m256 &other) const
{
        return _mm256_max_ps(data, other);
}

inline _m256 _m256::min(const _m256 &other) const
{
        return _mm256_min_ps(data, other);
}

inline float _m256::sum() const
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

inline _m256 _mask8::operator() (const _m256 &first, const _m256 &second) const
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

} // namespace nn::simd
