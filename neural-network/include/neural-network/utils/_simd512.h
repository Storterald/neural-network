#pragma once

#ifndef TARGET_X86_64
#error utils/_simd512.h cannot be included if the architecture is not x86_64
#endif // !TARGET_X86_64

#if SIMD_SUPPORT_LEVEL < 3
#error utils/_simd512.h cannot be included if the CPU does not support AVX512
#endif // SIMD_SUPPORT_LEVEL < 3

#include <immintrin.h>

#include <cstdint>

namespace nn::simd {

class _m512;
class _mask16 {
public:
        static constexpr __mmask16 ones = 0xFFFF;

        inline _mask16(__mmask16 mask) : data(mask) {}
        inline operator __mmask16() const { return data; }

        [[nodiscard]] inline _m512 operator() (const _m512 &first, const _m512 &second) const;

private:
        __mmask16 data;

}; // _mask16

class _m512 {
public:
        static constexpr uint32_t width     = 16;
        using mask                          = _mask16;

        inline _m512() : data(_mm512_setzero_ps()) {}
        inline _m512(const float *values) : data(_mm512_loadu_ps(values)) {}
        inline _m512(float scalar) : data(_mm512_set1_ps(scalar)) {}
        inline _m512(const __m512 &reg) : data(reg) {}
        inline operator __m512() const { return data; }

        inline void store(float *dst) const;

        [[nodiscard]] inline _m512 operator& (const _m512 &other) const;
        [[nodiscard]] inline _m512 operator| (const _m512 &other) const;
        [[nodiscard]] inline _m512 operator^ (const _m512 &other) const;

        [[nodiscard]] inline _m512 operator+ (const _m512 &other) const;
        [[nodiscard]] inline _m512 operator- (const _m512 &other) const;
        [[nodiscard]] inline _m512 operator* (const _m512 &other) const;
        [[nodiscard]] inline _m512 operator/ (const _m512 &other) const;

        inline _m512 &operator+= (const _m512 &other);
        inline _m512 &operator-= (const _m512 &other);
        inline _m512 &operator*= (const _m512 &other);
        inline _m512 &operator/= (const _m512 &other);

        [[nodiscard]] inline mask operator> (const _m512 &other) const;
        [[nodiscard]] inline mask operator< (const _m512 &other) const;
        [[nodiscard]] inline mask operator>= (const _m512 &other) const;
        [[nodiscard]] inline mask operator<= (const _m512 &other) const;
        [[nodiscard]] inline mask operator== (const _m512 &other) const;
        [[nodiscard]] inline mask operator!= (const _m512 &other) const;

        [[nodiscard]] inline _m512 abs() const;
        [[nodiscard]] inline _m512 fma(const _m512 &b, const _m512 &c) const;
        [[nodiscard]] inline _m512 fnma(const _m512 &b, const _m512 &c) const;
        [[nodiscard]] inline _m512 max(const _m512 &other) const;
        [[nodiscard]] inline _m512 min(const _m512 &other) const;

        [[nodiscard]] inline float sum() const;

private:
        __m512 data;

}; // class _m512

inline void _m512::store(float *dst) const
{
        _mm512_storeu_ps(dst, data);
}

inline _m512 _m512::operator& (const _m512 &other) const
{
        return _mm512_and_ps(data, other);
}

inline _m512 _m512::operator| (const _m512 &other) const
{
        return _mm512_or_ps(data, other);
}

inline _m512 _m512::operator^ (const _m512 &other) const
{
        return _mm512_xor_ps(data, other);
}

inline _m512 _m512::operator+ (const _m512 &other) const
{
        return _mm512_add_ps(data, other);
}

inline _m512 _m512::operator- (const _m512 &other) const
{
        return _mm512_sub_ps(data, other);
}

inline _m512 _m512::operator* (const _m512 &other) const
{
        return _mm512_mul_ps(data, other);
}

inline _m512 _m512::operator/ (const _m512 &other) const
{
        return _mm512_div_ps(data, other);
}

inline _m512 &_m512::operator+= (const _m512 &other)
{
        data = _mm512_add_ps(data, other);
        return *this;
}

inline _m512 &_m512::operator-= (const _m512 &other)
{
        data = _mm512_sub_ps(data, other);
        return *this;
}

inline _m512 &_m512::operator*= (const _m512 &other)
{
        data = _mm512_mul_ps(data, other);
        return *this;
}

inline _m512 &_m512::operator/= (const _m512 &other)
{
        data = _mm512_div_ps(data, other);
        return *this;
}

inline _m512::mask _m512::operator> (const _m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_GT_OQ);
}

inline _m512::mask _m512::operator< (const _m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_LT_OQ);
}

inline _m512::mask _m512::operator>= (const _m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_GE_OQ);
}

inline _m512::mask _m512::operator<= (const _m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_LE_OQ);
}

inline _m512::mask _m512::operator== (const _m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_EQ_OQ);
}

inline _m512::mask _m512::operator!= (const _m512 &other) const
{
        return _mm512_cmp_ps_mask(data, other, _CMP_NEQ_OQ);
}

inline _m512 _m512::abs() const
{
        return _mm512_abs_ps(data);
}

inline _m512 _m512::fma(const _m512 &b, const _m512 &c) const
{
        return _mm512_fmadd_ps(data, b, c);
}

inline _m512 _m512::fnma(const _m512 &b, const _m512 &c) const
{
        return _mm512_fnmadd_ps(data, b, c);
}

inline _m512 _m512::max(const _m512 &other) const
{
        return _mm512_max_ps(data, other);
}

inline _m512 _m512::min(const _m512 &other) const
{
        return _mm512_min_ps(data, other);
}

inline float _m512::sum() const
{
        return _mm512_reduce_add_ps(data);
}

inline _m512 _mask16::operator() (const _m512 &first, const _m512 &second) const
{
        return _mm512_mask_blend_ps(data, first, second);
}

} // namespace nn::simd
