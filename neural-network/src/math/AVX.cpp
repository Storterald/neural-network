#include "_math.h"

#include <neural-network/base.h>

#ifdef IS_X86_64BIT

#include <immintrin.h>

#include <cstdint>

#ifdef _mm256_cmp_ps_mask
#undef _mm256_cmp_ps_mask
#endif // _mm256_cmp_ps_mask

#define _mm256_abs_ps(X) (_mm256_and_ps(X, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))))
#define _mm256_cmp_ps_mask(a, b, cmp) ((__mmask8)_mm256_movemask_ps(_mm256_cmp_ps(a, b, cmp)))

static inline float _mm256_reduce_add_ps(__m256 v)
{
        const __m128 low    = _mm256_castps256_ps128(v);
        const __m128 high   = _mm256_extractf128_ps(v, 1);
        const __m128 sum128 = _mm_add_ps(low, high);
        __m128 shuf         = _mm_movehdup_ps(sum128);
        __m128 sums         = _mm_add_ps(sum128, shuf);
        shuf                = _mm_movehl_ps(shuf, sums);
        sums                = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

static inline __m256 __mm256_mask_blend_ps(__mmask8 k, __m256 a, __m256 b)
{
        const __m256i blend_mask = _mm256_set_epi32(
               (k & 0x80) * -1,
               (k & 0x40) * -1,
               (k & 0x20) * -1,
               (k & 0x10) * -1,
               (k & 0x08) * -1,
               (k & 0x04) * -1,
               (k & 0x02) * -1,
               (k & 0x01) * -1);

        const __m256 blend_mask_ps = _mm256_castsi256_ps(blend_mask);

        return _mm256_or_ps(
            _mm256_andnot_ps(blend_mask_ps, a),
            _mm256_and_ps(blend_mask_ps, b)
        );
}

#define _mm256_mask_blend_ps __mm256_mask_blend_ps

namespace nn {

template<> void _math<MATH_AVX>::sum(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);
                const __m256 sumResult   = _mm256_add_ps(values, otherValues);

                _mm256_storeu_ps(&result[i], sumResult);
        }

        _math<MATH_NORMAL>::sum(size - end, first + end, second + end, result + end);
}

template<> void _math<MATH_AVX>::sub(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);
                const __m256 subResult   = _mm256_sub_ps(values, otherValues);

                _mm256_storeu_ps(&result[i], subResult);
        }

        _math<MATH_NORMAL>::sub(size - end, first + end, second + end, result + end);
}

template<> void _math<MATH_AVX>::mul(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);
                const __m256 mulResult   = _mm256_mul_ps(values, otherValues);

                _mm256_storeu_ps(&result[i], mulResult);
        }

        _math<MATH_NORMAL>::mul(size - end, first + end, second + end, result + end);
}

template<> void _math<MATH_AVX>::div(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);
                const __m256 divResult   = _mm256_div_ps(values, otherValues);

                _mm256_storeu_ps(&result[i], divResult);
        }

        _math<MATH_NORMAL>::div(size - end, first + end, second + end, result + end);
}

template<> void _math<MATH_AVX>::sum(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 scalarValues = _mm256_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 sumResult = _mm256_add_ps(values, scalarValues);

                _mm256_storeu_ps(&result[i], sumResult);
        }

        _math<MATH_NORMAL>::sum(size - end, data + end, scalar, result + end);
}

template<> void _math<MATH_AVX>::sub(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 scalarValues = _mm256_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 subResult = _mm256_sub_ps(values, scalarValues);

                _mm256_storeu_ps(&result[i], subResult);
        }

        _math<MATH_NORMAL>::sub(size - end, data + end, scalar, result + end);
}

template<> void _math<MATH_AVX>::mul(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 scalarValues = _mm256_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 mulResult = _mm256_mul_ps(values, scalarValues);

                _mm256_storeu_ps(&result[i], mulResult);
        }

        _math<MATH_NORMAL>::mul(size - end, data + end, scalar, result + end);
}

template<> void _math<MATH_AVX>::div(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 scalarValues = _mm256_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 divResult = _mm256_div_ps(values, scalarValues);

                _mm256_storeu_ps(&result[i], divResult);
        }

        _math<MATH_NORMAL>::div(size - end, data + end, scalar, result + end);
}

template<> void _math<MATH_AVX>::tanh(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 threshold = _mm256_set1_ps(4.9f);
        const __m256 one       = _mm256_set1_ps(1.0f);
        const __m256 negative  = _mm256_set1_ps(-0.0f);

        const __m256 v28     = _mm256_set1_ps(28.0f);
        const __m256 v378    = _mm256_set1_ps(378.0f);
        const __m256 v3150   = _mm256_set1_ps(3150.0f);
        const __m256 v17325  = _mm256_set1_ps(17325.0f);
        const __m256 v62370  = _mm256_set1_ps(62370.0f);
        const __m256 v135135 = _mm256_set1_ps(135135.0f);

        for (uint32_t i = 0; i < end; i += SIMD_WIDTH) {
                const __m256 x  = _mm256_loadu_ps(&data[i]);
                const __m256 x2 = _mm256_mul_ps(x, x);

                const __m256 absoluteX  = _mm256_abs_ps(x);
                const __mmask8 mask     = _mm256_cmp_ps_mask(absoluteX, threshold, _CMP_GE_OQ);
                const __m256 signs      = _mm256_and_ps(x, negative);
                const __m256 signed_one = _mm256_or_ps(signs, one);

                __m256 numerator = _mm256_add_ps(x2, v378);
                numerator        = _mm256_fmadd_ps(x2, numerator, v17325);
                numerator        = _mm256_fmadd_ps(x2, numerator, v135135);
                numerator        = _mm256_mul_ps(x, numerator);

                __m256 denominator = _mm256_fmadd_ps(x2, v28, v3150);
                denominator        = _mm256_fmadd_ps(x2, denominator, v62370);
                denominator        = _mm256_fmadd_ps(x2, denominator, v135135);

                __m256 tanh = _mm256_div_ps(numerator, denominator);
                tanh        = _mm256_mask_blend_ps(mask, tanh, signed_one);

                _mm256_storeu_ps(&result[i], tanh);
        }

        _math<MATH_NORMAL>::tanh(size - end, data + end, result + end);
}

template<> void _math<MATH_AVX>::tanh_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        _math<MATH_AVX>::tanh(size - end, data, result);

        const __m256 threshold = _mm256_set1_ps(4.9f);
        const __m256 zero      = _mm256_setzero_ps();
        const __m256 one       = _mm256_set1_ps(1.0f);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values     = _mm256_loadu_ps(&data[i]);
                const __m256 tanhValues = _mm256_loadu_ps(&result[i]);
                __m256 tanhDerivative   = _mm256_fnmadd_ps(tanhValues, tanhValues, one);

                const __mmask8 mask_large = _mm256_cmp_ps_mask(_mm256_abs_ps(values), threshold, _CMP_GT_OQ);
                tanhDerivative            = _mm256_mask_blend_ps(mask_large, tanhDerivative, zero);
                const __mmask8 mask_zero  = _mm256_cmp_ps_mask(values, zero, _CMP_EQ_OQ);
                tanhDerivative            = _mm256_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm256_storeu_ps(&result[i], tanhDerivative);
        }

        _math<MATH_NORMAL>::tanh_derivative(size - end, data + end, result + end);
}

template<> void _math<MATH_AVX>::ReLU(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 zero = _mm256_setzero_ps();

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 x    = _mm256_loadu_ps(&data[i]);
                const __m256 relu = _mm256_max_ps(zero, x);

                _mm256_storeu_ps(&result[i], relu);
        }

        _math<MATH_NORMAL>::ReLU(size - end, data + end, result + end);
}

template<> void _math<MATH_AVX>::ReLU_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 zero = _mm256_setzero_ps();
        const __m256 one  = _mm256_set1_ps(1.0f);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 x              = _mm256_loadu_ps(&data[i]);
                const __mmask8 mask         = _mm256_cmp_ps_mask(x, zero, _CMP_GT_OQ);
                const __m256 reluDerivative = _mm256_mask_blend_ps(mask, zero, one);

                _mm256_storeu_ps(&result[i], reluDerivative);
        }

        _math<MATH_NORMAL>::ReLU_derivative(size - end, data + end, result + end);
}

template<> void _math<MATH_AVX>::min(
        uint32_t           size,
        const float        data[],
        float              min,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 minValues = _mm256_set1_ps(min);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 minResult = _mm256_min_ps(values, minValues);

                _mm256_storeu_ps(&result[i], minResult);
        }

        _math<MATH_NORMAL>::min(size - end, data + end, min, result + end);
}


template<> void _math<MATH_AVX>::max(
        uint32_t           size,
        const float        data[],
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 maxValues = _mm256_set1_ps(max);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 maxResult = _mm256_max_ps(values, maxValues);

                _mm256_storeu_ps(&result[i], maxResult);
        }

        _math<MATH_NORMAL>::max(size - end, data + end, max, result + end);
}

template<> void _math<MATH_AVX>::clamp(
        uint32_t           size,
        const float        data[],
        float              min,
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m256 minValues = _mm256_set1_ps(min);
        const __m256 maxValues = _mm256_set1_ps(max);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values = _mm256_loadu_ps(&data[i]);
                const __m256 clamp  = _mm256_max_ps(_mm256_min_ps(maxValues, values), minValues);

                _mm256_storeu_ps(&result[i], clamp);
        }

        _math<MATH_NORMAL>::clamp(size - end, data + end, min, max, result + end);
}

template<> void _math<MATH_AVX>::min(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);
                const __m256 minValues   = _mm256_min_ps(otherValues, values);

                _mm256_storeu_ps(&result[i], minValues);
        }

        _math<MATH_NORMAL>::min(size - end, first + end, second + end, result + end);
}

template<> void _math<MATH_AVX>::max(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);
                const __m256 maxValues   = _mm256_max_ps(otherValues, values);

                _mm256_storeu_ps(&result[i], maxValues);
        }

        _math<MATH_NORMAL>::max(size - end, first + end, second + end, result + end);
}

template<> void _math<MATH_AVX>::clamp(
        uint32_t           size,
        const float        data[],
        const float        min[],
        const float        max[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values    = _mm256_loadu_ps(&data[i]);
                const __m256 minValues = _mm256_loadu_ps(&min[i]);
                const __m256 maxValues = _mm256_loadu_ps(&max[i]);
                const __m256 clamp     = _mm256_max_ps(_mm256_min_ps(maxValues, values), minValues);

                _mm256_storeu_ps(&result[i], clamp);
        }

        _math<MATH_NORMAL>::clamp(size - end, data + end, min + end, max + end, result + end);
}

template<> void _math<MATH_AVX>::compare(
        uint32_t           size,
        const float        first[],
        const float        second[],
        bool               *result) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        *result = true;

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m256 values      = _mm256_loadu_ps(&first[i]);
                const __m256 otherValues = _mm256_loadu_ps(&second[i]);

                const __mmask8 cmp = _mm256_cmp_ps_mask(values, otherValues, _CMP_EQ_OS);
                if (cmp != 0xFF) {
                        *result = false;
                        return;
                }
        }

        _math<MATH_NORMAL>::compare(size - end, first + end, second + end, result);
}

template<> void _math<MATH_AVX>::matvec_mul(
        uint32_t           width,
        uint32_t           height,
        const float        matrix[],
        const float        vector[],
        float              result[]) {

        const uint32_t end = width & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < height; ++i) {
                __m256 sum = _mm256_setzero_ps();

                for (uint32_t j = 0; j < end; j+=SIMD_WIDTH) {
                        const __m256 values       = _mm256_loadu_ps(&matrix[i * width + j]);
                        const __m256 vectorValues = _mm256_loadu_ps(&vector[j]);
                        const __m256 product      = _mm256_mul_ps(values, vectorValues);

                        sum = _mm256_add_ps(sum, product);
                }

                result[i] = _mm256_reduce_add_ps(sum);

                for (uint32_t j = end; j < width; ++j)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}

} // namespace nn

#endif // IS_X86_64BIT
