#include <neural-network/math/_Math.h>

#include <immintrin.h>

#include <cstdint>

template<> void _Math<MATH_AVX512>::sum(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);
                const __m512 sumResult   = _mm512_add_ps(values, otherValues);

                _mm512_storeu_ps(&result[i], sumResult);
        }

        _Math<MATH_AVX>::sum(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_AVX512>::sub(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);
                const __m512 subResult   = _mm512_sub_ps(values, otherValues);

                _mm512_storeu_ps(&result[i], subResult);
        }

        _Math<MATH_AVX>::sub(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_AVX512>::mul(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);
                const __m512 mulResult   = _mm512_mul_ps(values, otherValues);

                _mm512_storeu_ps(&result[i], mulResult);
        }

        _Math<MATH_AVX>::mul(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_AVX512>::div(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);
                const __m512 divResult   = _mm512_div_ps(values, otherValues);

                _mm512_storeu_ps(&result[i], divResult);
        }

        _Math<MATH_AVX>::div(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_AVX512>::sum(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 scalarValues = _mm512_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 sumResult = _mm512_add_ps(values, scalarValues);

                _mm512_storeu_ps(&result[i], sumResult);
        }

        _Math<MATH_AVX>::sum(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_AVX512>::sub(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 scalarValues = _mm512_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 subResult = _mm512_sub_ps(values, scalarValues);

                _mm512_storeu_ps(&result[i], subResult);
        }

        _Math<MATH_AVX>::sub(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_AVX512>::mul(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 scalarValues = _mm512_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 mulResult = _mm512_mul_ps(values, scalarValues);

                _mm512_storeu_ps(&result[i], mulResult);
        }

        _Math<MATH_AVX>::mul(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_AVX512>::div(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 scalarValues = _mm512_set1_ps(scalar);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 divResult = _mm512_div_ps(values, scalarValues);

                _mm512_storeu_ps(&result[i], divResult);
        }

        _Math<MATH_AVX>::div(size - end, data + end, scalar, result + end);
}

template<> void _Math<MATH_AVX512>::tanh(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 threshold = _mm512_set1_ps(4.9f);
        const __m512 one       = _mm512_set1_ps(1.0f);
        const __m512 negative  = _mm512_set1_ps(-0.0f);

        const __m512 v28     = _mm512_set1_ps(28.0f);
        const __m512 v378    = _mm512_set1_ps(378.0f);
        const __m512 v3150   = _mm512_set1_ps(3150.0f);
        const __m512 v17325  = _mm512_set1_ps(17325.0f);
        const __m512 v62370  = _mm512_set1_ps(62370.0f);
        const __m512 v135135 = _mm512_set1_ps(135135.0f);

        for (uint32_t i = 0; i < end; i += SIMD_WIDTH) {
                const __m512 x  = _mm512_loadu_ps(&data[i]);
                const __m512 x2 = _mm512_mul_ps(x, x);

                const __m512 absoluteX  = _mm512_abs_ps(x);
                const __mmask16 mask    = _mm512_cmp_ps_mask(absoluteX, threshold, _CMP_GE_OQ);
                const __m512 signs      = _mm512_and_ps(x, negative);
                const __m512 signed_one = _mm512_or_ps(signs, one);

                __m512 numerator = _mm512_add_ps(x2, v378);
                numerator        = _mm512_fmadd_ps(x2, numerator, v17325);
                numerator        = _mm512_fmadd_ps(x2, numerator, v135135);
                numerator        = _mm512_mul_ps(x, numerator);

                __m512 denominator = _mm512_fmadd_ps(x2, v28, v3150);
                denominator        = _mm512_fmadd_ps(x2, denominator, v62370);
                denominator        = _mm512_fmadd_ps(x2, denominator, v135135);

                __m512 tanh = _mm512_div_ps(numerator, denominator);
                tanh        = _mm512_mask_blend_ps(mask, tanh, signed_one);

                _mm512_storeu_ps(&result[i], tanh);
        }

        _Math<MATH_AVX>::tanh(size - end, data + end, result + end);
}

template<> void _Math<MATH_AVX512>::tanh_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        _Math<MATH_AVX512>::tanh(size - end, data, result);

        const __m512 threshold = _mm512_set1_ps(4.9f);
        const __m512 zero      = _mm512_setzero_ps();
        const __m512 one       = _mm512_set1_ps(1.0f);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values     = _mm512_loadu_ps(&data[i]);
                const __m512 tanhValues = _mm512_loadu_ps(&result[i]);
                __m512 tanhDerivative   = _mm512_fnmadd_ps(tanhValues, tanhValues, one);

                const __mmask16 mask_large = _mm512_cmp_ps_mask(_mm512_abs_ps(values), threshold, _CMP_GT_OQ);
                tanhDerivative             = _mm512_mask_blend_ps(mask_large, tanhDerivative, zero);

                const __mmask16 mask_zero = _mm512_cmp_ps_mask(values, zero, _CMP_EQ_OQ);
                tanhDerivative            = _mm512_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm512_storeu_ps(&result[i], tanhDerivative);
        }

        _Math<MATH_AVX>::tanh_derivative(size - end, data + end, result + end);
}

template<> void _Math<MATH_AVX512>::ReLU(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 zero = _mm512_setzero_ps();

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 x    = _mm512_loadu_ps(&data[i]);
                const __m512 relu = _mm512_max_ps(zero, x);

                _mm512_storeu_ps(&result[i], relu);
        }

        _Math<MATH_AVX>::ReLU(size - end, data + end, result + end);
}

template<> void _Math<MATH_AVX512>::ReLU_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 zero = _mm512_setzero_ps();
        const __m512 one  = _mm512_set1_ps(1.0f);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 x              = _mm512_loadu_ps(&data[i]);
                const __mmask16 mask        = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
                const __m512 reluDerivative = _mm512_mask_blend_ps(mask, zero, one);

                _mm512_storeu_ps(&result[i], reluDerivative);
        }

        _Math<MATH_AVX>::ReLU_derivative(size - end, data + end, result + end);
}

template<> void _Math<MATH_AVX512>::min(
        uint32_t           size,
        const float        data[],
        float              min,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 minValues = _mm512_set1_ps(min);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 minResult = _mm512_min_ps(values, minValues);

                _mm512_storeu_ps(&result[i], minResult);
        }

        _Math<MATH_AVX>::min(size - end, data + end, min, result + end);
}

template<> void _Math<MATH_AVX512>::max(
        uint32_t           size,
        const float        data[],
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 maxValues = _mm512_set1_ps(max);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 maxResult = _mm512_max_ps(values, maxValues);

                _mm512_storeu_ps(&result[i], maxResult);
        }

        _Math<MATH_AVX>::max(size - end, data + end, max, result + end);
}

template<> void _Math<MATH_AVX512>::clamp(
        uint32_t           size,
        const float        data[],
        float              min,
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        const __m512 minValues = _mm512_set1_ps(min);
        const __m512 maxValues = _mm512_set1_ps(max);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values = _mm512_loadu_ps(&data[i]);
                const __m512 clamp  = _mm512_max_ps(_mm512_min_ps(maxValues, values), minValues);

                _mm512_storeu_ps(&result[i], clamp);
        }

        _Math<MATH_AVX>::clamp(size - end, data + end, min, max, result + end);
}

template<> void _Math<MATH_AVX512>::min(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);
                const __m512 minValues   = _mm512_min_ps(otherValues, values);

                _mm512_storeu_ps(&result[i], minValues);
        }

        _Math<MATH_AVX>::min(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_AVX512>::max(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);
                const __m512 maxValues   = _mm512_max_ps(otherValues, values);

                _mm512_storeu_ps(&result[i], maxValues);
        }

        _Math<MATH_AVX>::max(size - end, first + end, second + end, result + end);
}

template<> void _Math<MATH_AVX512>::clamp(
        uint32_t           size,
        const float        data[],
        const float        min[],
        const float        max[],
        float              result[]) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values    = _mm512_loadu_ps(&data[i]);
                const __m512 minValues = _mm512_loadu_ps(&min[i]);
                const __m512 maxValues = _mm512_loadu_ps(&max[i]);
                const __m512 clamp     = _mm512_max_ps(_mm512_min_ps(maxValues, values), minValues);

                _mm512_storeu_ps(&result[i], clamp);
        }

        _Math<MATH_AVX>::clamp(size - end, data + end, min + end, max + end, result + end);
}

template<> void _Math<MATH_AVX512>::compare(
        uint32_t           size,
        const float        first[],
        const float        second[],
        bool               *result) {

        const uint32_t end = size & ~(SIMD_WIDTH - 1);

        *result = false;

        for (uint32_t i = 0; i < end; i+=SIMD_WIDTH) {
                const __m512 values      = _mm512_loadu_ps(&first[i]);
                const __m512 otherValues = _mm512_loadu_ps(&second[i]);

                const __mmask16 cmp = _mm512_cmp_ps_mask(values, otherValues, _CMP_EQ_OS);
                if (cmp != 0xFFFF) {
                        *result = false;
                        return;
                }
        }

        _Math<MATH_AVX>::compare(size - end, first + end, second + end, result);
}

template<> void _Math<MATH_AVX512>::matvec_mul(
        uint32_t           width,
        uint32_t           height,
        const float        matrix[],
        const float        vector[],
        float              result[]) {

        const uint32_t end = width & ~(SIMD_WIDTH - 1);

        for (uint32_t i = 0; i < height; ++i) {
                __m512 sum = _mm512_setzero_ps();

                for (uint32_t j = 0; j < end; j+=SIMD_WIDTH) {
                        const __m512 values       = _mm512_loadu_ps(&matrix[i * width + j]);
                        const __m512 vectorValues = _mm512_loadu_ps(&vector[j]);
                        const __m512 product      = _mm512_mul_ps(values, vectorValues);

                        sum = _mm512_add_ps(sum, product);
                }

                result[i] = _mm512_reduce_add_ps(sum);

                for (uint32_t j = end; j < width; ++j)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}
