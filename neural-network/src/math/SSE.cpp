#include "_Math.h"

#include <immintrin.h>

// An equivalent of _mm512_abs_ps for SSE does not exists, so this just
// changes the most significant bit of the numbers to 0 (positive).
#define _mm_abs_ps(X) _mm_and_ps(X, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)))

// An equivalent of _mm512_reduce_add_ps for SSE does not exists.
static inline float _reduce_add_ps_sse(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 sums = _mm_add_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, sums);
        sums        = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

#define _mm_reduce_add_ps(X) _reduce_add_ps_sse(X)

template<> void _Math<MATH_SSE>::sum(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values      = _mm_loadu_ps(&first[i]);
                const __m128 otherValues = _mm_loadu_ps(&second[i]);
                const __m128 sumResult   = _mm_add_ps(values, otherValues);

                _mm_storeu_ps(&result[i], sumResult);
        }

        _Math<MATH_NORMAL>::sum(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_SSE>::sub(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values      = _mm_loadu_ps(&first[i]);
                const __m128 otherValues = _mm_loadu_ps(&second[i]);
                const __m128 subResult   = _mm_sub_ps(values, otherValues);

                _mm_storeu_ps(&result[i], subResult);
        }

        _Math<MATH_NORMAL>::sub(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_SSE>::mul(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values      = _mm_loadu_ps(&first[i]);
                const __m128 otherValues = _mm_loadu_ps(&second[i]);
                const __m128 mulResult   = _mm_mul_ps(values, otherValues);

                _mm_storeu_ps(&result[i], mulResult);
        }

        _Math<MATH_NORMAL>::mul(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_SSE>::div(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values      = _mm_loadu_ps(&first[i]);
                const __m128 otherValues = _mm_loadu_ps(&second[i]);
                const __m128 divResult   = _mm_div_ps(values, otherValues);

                _mm_storeu_ps(&result[i], divResult);
        }

        _Math<MATH_NORMAL>::div(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_SSE>::sum(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 scalarValues = _mm_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 sumResult = _mm_add_ps(values, scalarValues);

                _mm_storeu_ps(&result[i], sumResult);
        }

        _Math<MATH_NORMAL>::sum(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_SSE>::sub(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 scalarValues = _mm_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 subResult = _mm_sub_ps(values, scalarValues);

                _mm_storeu_ps(&result[i], subResult);
        }

        _Math<MATH_NORMAL>::sub(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_SSE>::mul(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 scalarValues = _mm_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 mulResult = _mm_mul_ps(values, scalarValues);

                _mm_storeu_ps(&result[i], mulResult);
        }

        _Math<MATH_NORMAL>::mul(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_SSE>::div(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 scalarValues = _mm_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 divResult = _mm_div_ps(values, scalarValues);

                _mm_storeu_ps(&result[i], divResult);
        }

        _Math<MATH_NORMAL>::div(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_SSE>::tanh(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        // Constant values for mask
        const __m128 threshold = _mm_set1_ps(4.9f);
        const __m128 one       = _mm_set1_ps(1.0f);
        const __m128 negative  = _mm_set1_ps(-0.0f);

        // Constant values for approximation
        const __m128 v28     = _mm_set1_ps(28.0f);
        const __m128 v378    = _mm_set1_ps(378.0f);
        const __m128 v3150   = _mm_set1_ps(3150.0f);
        const __m128 v17325  = _mm_set1_ps(17325.0f);
        const __m128 v62370  = _mm_set1_ps(62370.0f);
        const __m128 v135135 = _mm_set1_ps(135135.0f);

        for (uint32_t i = 0; i < end; i += SIMD_WIDTH) {
                const __m128 x  = _mm_loadu_ps(&data[i]);
                const __m128 x2 = _mm_mul_ps(x, x);

                // Check if |x| >= 6.0
                const __m128 absoluteX  = _mm_abs_ps(x);
                const __mmask8 mask     = _mm_cmp_ps_mask(absoluteX, threshold, _CMP_GE_OQ);
                const __m128 signs      = _mm_and_ps(x, negative);
                const __m128 signed_one = _mm_or_ps(signs, one);

                // Numerator: x * (135135 + x2 * (17325 + x2 * (378 + x2)))
                __m128 numerator = _mm_add_ps(x2, v378);
                numerator        = _mm_fmadd_ps(x2, numerator, v17325);
                numerator        = _mm_fmadd_ps(x2, numerator, v135135);
                numerator        = _mm_mul_ps(x, numerator);

                // Denominator: 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
                __m128 denominator = _mm_fmadd_ps(x2, v28, v3150);
                denominator        = _mm_fmadd_ps(x2, denominator, v62370);
                denominator        = _mm_fmadd_ps(x2, denominator, v135135);

                __m128 tanh = _mm_div_ps(numerator, denominator);
                tanh        = _mm_mask_blend_ps(mask, tanh, signed_one);

                _mm_storeu_ps(&result[i], tanh);
        }

        _Math<MATH_NORMAL>::tanh(size - end, data + end, result + end);
}

template<> void _Math<MATH_SSE>::tanh_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        // Tanh values are stored in the result array, then overwritten.
        _Math<MATH_SSE>::tanh(size, data, result);

        const __m128 threshold = _mm_set1_ps(4.9f);
        const __m128 zero      = _mm_setzero_ps();
        const __m128 one       = _mm_set1_ps(1.0f);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values     = _mm_loadu_ps(&data[i]);
                const __m128 tanhValues = _mm_loadu_ps(&result[i]);
                __m128 tanhDerivative   = _mm_fnmadd_ps(tanhValues, tanhValues, one);

                const __mmask8 mask_large = _mm_cmp_ps_mask(_mm_abs_ps(values), threshold, _CMP_GT_OQ);
                tanhDerivative            = _mm_mask_blend_ps(mask_large, tanhDerivative, zero);

                const __mmask8 mask_zero = _mm_cmp_ps_mask(values, zero, _CMP_EQ_OQ);
                tanhDerivative           = _mm_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm_storeu_ps(&result[i], tanhDerivative);
        }

        _Math<MATH_NORMAL>::tanh_derivative(size - end, data + end, result + end);
}

template<> void _Math<MATH_SSE>::ReLU(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 zero = _mm_setzero_ps();
        
        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 x    = _mm_loadu_ps(&data[i]);
                const __m128 relu = _mm_max_ps(zero, x);

                _mm_storeu_ps(&result[i], relu);
        }

        _Math<MATH_NORMAL>::ReLU(size - end, data + end, result + end);
}

template<> void _Math<MATH_SSE>::ReLU_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 zero = _mm_setzero_ps();
        const __m128 one  = _mm_set1_ps(1.0f);
        
        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 x              = _mm_loadu_ps(&data[i]);
                const __mmask8 mask         = _mm_cmp_ps_mask(x, zero, _CMP_GT_OQ);
                const __m128 reluDerivative = _mm_mask_blend_ps(mask, zero, one);

                _mm_storeu_ps(&result[i], reluDerivative);
        }

        _Math<MATH_NORMAL>::ReLU_derivative(size - end, data + end, result + end);
}

template<> void _Math<MATH_SSE>::min(
        uint32_t           size,
        const float        data[],
        float              min,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 minValues = _mm_set1_ps(min);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 minResult = _mm_min_ps(values, minValues);

                _mm_storeu_ps(&result[i], minResult);
        }

        _Math<MATH_NORMAL>::min(size - end, data + end, min, result + end);
}

template<> void _Math<MATH_SSE>::max(
        uint32_t           size,
        const float        data[],
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 maxValues = _mm_set1_ps(max);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 maxResult = _mm_max_ps(values, maxValues);

                _mm_storeu_ps(&result[i], maxResult);
        }

        _Math<MATH_NORMAL>::max(size - end, data + end, max, result + end);
}

template<> void _Math<MATH_SSE>::clamp(
        uint32_t           size,
        const float        data[],
        float              min,
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m128 minValues = _mm_set1_ps(min);
        const __m128 maxValues = _mm_set1_ps(max);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values = _mm_loadu_ps(&data[i]);
                const __m128 clamp  = _mm_max_ps(_mm_min_ps(maxValues, values), minValues);

                _mm_storeu_ps(&result[i], clamp);
        }

        _Math<MATH_NORMAL>::clamp(size - end, data + end, min, max, result + end);
}

template<> void _Math<MATH_SSE>::min(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values      = _mm_loadu_ps(&first[i]);
                const __m128 otherValues = _mm_loadu_ps(&second[i]);
                const __m128 minValues   = _mm_min_ps(otherValues, values);

                _mm_storeu_ps(&result[i], minValues);
        }

        _Math<MATH_NORMAL>::min(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_SSE>::max(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values      = _mm_loadu_ps(&first[i]);
                const __m128 otherValues = _mm_loadu_ps(&second[i]);
                const __m128 maxValues   = _mm_max_ps(otherValues, values);

                _mm_storeu_ps(&result[i], maxValues);
        }

        _Math<MATH_NORMAL>::max(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_SSE>::clamp(
        uint32_t           size,
        const float        data[],
        const float        min[],
        const float        max[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m128 values    = _mm_loadu_ps(&data[i]);
                const __m128 minValues = _mm_loadu_ps(&min[i]);
                const __m128 maxValues = _mm_loadu_ps(&max[i]);
                const __m128 clamp     = _mm_max_ps(_mm_min_ps(maxValues, values), minValues);

                _mm_storeu_ps(&result[i], clamp);
        }

        _Math<MATH_NORMAL>::clamp(size - end, data + end, min + end, max + end, result + end);
}

template<> void _Math<MATH_SSE>::matvec_mul(
        uint32_t           width,
        uint32_t           height,
        const float        matrix[],
        const float        vector[],
        float              result[]) {

        const uint32_t end = width & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < height; i++) {
                __m128 sum = _mm_setzero_ps();

                for (uint32_t j = 0; j < end; j+=SIMD_WIDTH) {
                        const __m128 values       = _mm_loadu_ps(&matrix[i * width + j]);
                        const __m128 vectorValues = _mm_loadu_ps(&vector[j]);
                        const __m128 product      = _mm_mul_ps(values, vectorValues);

                        sum = _mm_add_ps(sum, product);
                }

                result[i] = _mm_reduce_add_ps(sum);

                for (uint32_t j = end; j < width; j++)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}