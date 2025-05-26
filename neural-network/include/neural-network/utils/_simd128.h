#pragma once

#ifndef TARGET_X86_64
#error utils/_simd128.h cannot be included if the architecture is not x86_64
#endif // !TARGET_X86_64

#if SIMD_SUPPORT_LEVEL < 1
#error utils/_simd128.h cannot be included if the CPU does not support SSE3
#endif // SIMD_SUPPORT_LEVEL < 2

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3

#include <cstdint>

#define _mm_cmp_ps__CMP_EQ_OQ  _mm_cmpeq_ps
#define _mm_cmp_ps__CMP_NEQ_OQ _mm_cmpneq_ps
#define _mm_cmp_ps__CMP_GE_OQ  _mm_cmpge_ps
#define _mm_cmp_ps__CMP_GT_OQ  _mm_cmpgt_ps
#define _mm_cmp_ps__CMP_LE_OQ  _mm_cmple_ps
#define _mm_cmp_ps__CMP_LT_OQ  _mm_cmplt_ps

#define _mm_cmp_ps_mask(a, b, cmp) _mm_movemask_ps(_mm_cmp_ps_##cmp(a, b))

namespace nn::simd {

class _m128;
class _mask4 {
public:
        static constexpr int ones = 0x0F;

        inline _mask4(int mask) : data(mask) {}
        inline operator int() const { return data; }

        [[nodiscard]] inline _m128 operator() (const _m128 &first, const _m128 &second) const;

private:
        int data;

}; // _mask4

class _m128 {
public:
        static constexpr uint32_t width     = 4;
        using mask                          = _mask4;

        inline _m128() : data(_mm_setzero_ps()) {}
        inline _m128(const float *values) : data(_mm_loadu_ps(values)) {}
        inline _m128(float scalar) : data(_mm_set1_ps(scalar)) {}
        inline _m128(const __m128 &reg) : data(reg) {}
        inline operator __m128() const { return data; }

        inline void store(float *dst) const;

        [[nodiscard]] inline _m128 operator& (const _m128 &other) const;
        [[nodiscard]] inline _m128 operator| (const _m128 &other) const;
        [[nodiscard]] inline _m128 operator^ (const _m128 &other) const;

        [[nodiscard]] inline _m128 operator+ (const _m128 &other) const;
        [[nodiscard]] inline _m128 operator- (const _m128 &other) const;
        [[nodiscard]] inline _m128 operator* (const _m128 &other) const;
        [[nodiscard]] inline _m128 operator/ (const _m128 &other) const;

        inline _m128 &operator+= (const _m128 &other);
        inline _m128 &operator-= (const _m128 &other);
        inline _m128 &operator*= (const _m128 &other);
        inline _m128 &operator/= (const _m128 &other);

        [[nodiscard]] inline mask operator> (const _m128 &other) const;
        [[nodiscard]] inline mask operator< (const _m128 &other) const;
        [[nodiscard]] inline mask operator>= (const _m128 &other) const;
        [[nodiscard]] inline mask operator<= (const _m128 &other) const;
        [[nodiscard]] inline mask operator== (const _m128 &other) const;
        [[nodiscard]] inline mask operator!= (const _m128 &other) const;

        [[nodiscard]] inline _m128 abs() const;
        [[nodiscard]] inline _m128 fma(const _m128 &b, const _m128 &c) const;
        [[nodiscard]] inline _m128 fnma(const _m128 &b, const _m128 &c) const;
        [[nodiscard]] inline _m128 max(const _m128 &other) const;
        [[nodiscard]] inline _m128 min(const _m128 &other) const;

        [[nodiscard]] inline float sum() const;

private:
        __m128 data;

}; // class _m128

inline void _m128::store(float *dst) const
{
        _mm_storeu_ps(dst, data);
}

inline _m128 _m128::operator& (const _m128 &other) const
{
        return _mm_and_ps(data, other);
}

inline _m128 _m128::operator| (const _m128 &other) const
{
        return _mm_or_ps(data, other);
}

inline _m128 _m128::operator^ (const _m128 &other) const
{
        return _mm_xor_ps(data, other);
}

inline _m128 _m128::operator+ (const _m128 &other) const
{
        return _mm_add_ps(data, other);
}

inline _m128 _m128::operator- (const _m128 &other) const
{
        return _mm_sub_ps(data, other);
}

inline _m128 _m128::operator* (const _m128 &other) const
{
        return _mm_mul_ps(data, other);
}

inline _m128 _m128::operator/ (const _m128 &other) const
{
        return _mm_div_ps(data, other);
}

inline _m128 &_m128::operator+= (const _m128 &other)
{
        data = _mm_add_ps(data, other);
        return *this;
}

inline _m128 &_m128::operator-= (const _m128 &other)
{
        data = _mm_sub_ps(data, other);
        return *this;
}

inline _m128 &_m128::operator*= (const _m128 &other)
{
        data = _mm_mul_ps(data, other);
        return *this;
}

inline _m128 &_m128::operator/= (const _m128 &other)
{
        data = _mm_div_ps(data, other);
        return *this;
}

inline _m128::mask _m128::operator> (const _m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_GT_OQ);
}

inline _m128::mask _m128::operator< (const _m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_LT_OQ);
}

inline _m128::mask _m128::operator>= (const _m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_GE_OQ);
}

inline _m128::mask _m128::operator<= (const _m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_LE_OQ);
}

inline _m128::mask _m128::operator== (const _m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_EQ_OQ);
}

inline _m128::mask _m128::operator!= (const _m128 &other) const
{
        return _mm_cmp_ps_mask(data, other, _CMP_NEQ_OQ);
}

inline _m128 _m128::abs() const
{
        return _mm_and_ps(data, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}

inline _m128 _m128::fma(const _m128 &b, const _m128 &c) const
{
        return _mm_add_ps(_mm_mul_ps(data, b), c);
}

inline _m128 _m128::fnma(const _m128 &b, const _m128 &c) const
{
        return _mm_sub_ps(c, _mm_mul_ps(data, b));
}

inline _m128 _m128::max(const _m128 &other) const
{
        return _mm_max_ps(data, other);
}

inline _m128 _m128::min(const _m128 &other) const
{
        return _mm_min_ps(data, other);
}

inline float _m128::sum() const
{
        __m128 shuf = _mm_movehdup_ps(data);
        __m128 sums = _mm_add_ps(data, shuf);
        shuf        = _mm_movehl_ps(shuf, sums);
        sums        = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

inline _m128 _mask4::operator() (const _m128 &first, const _m128 &second) const
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

} // namespace nn::simd

#undef _mm_cmp_ps_mask
