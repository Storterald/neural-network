#include "../math/FastMath.h"

#if !defined DEBUG_MODE_ENABLED && !defined DISABLE_AVX512
#include <immintrin.h>
#endif

Vector Fast::relu(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = Fast::relu(vec[i]);

        return result;
#else
        const uint32_t END { (vec.size() / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(vec.size());

        const __m512 zero { _mm512_setzero_ps() };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 x { _mm512_loadu_ps(&vec[i]) };
                const __m512 relu { _mm512_max_ps(zero, x) };

                _mm512_storeu_ps(&result[i], relu);
        }

        for (uint32_t i { END }; i < vec.size(); i++)
                result[i] = Fast::relu(vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Fast::reluDerivative(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = Fast::reluDerivative(vec[i]);

        return result;
#else
        const uint32_t END { (vec.size() / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(vec.size());

        const __m512 zero { _mm512_setzero_ps() };
        const __m512 one { _mm512_set1_ps(1.0f) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 x { _mm512_loadu_ps(&vec[i]) };
                const __mmask16 mask { _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) }; // Compare x > 0
                const __m512 reluDerivative { _mm512_mask_blend_ps(mask, zero, one) };

                _mm512_storeu_ps(&result[i], reluDerivative);
        }

        for (uint32_t i { END }; i < vec.size(); i++)
                result[i] = Fast::reluDerivative(vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Fast::tanh(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (std::size_t i { 0 }; i < vec.size(); i++)
                result[i] = Fast::tanh(vec[i]);

        return result;
#else
        const uint32_t END { (vec.size() / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(vec.size());

        // Constant values for mask
        const __m512 threshold { _mm512_set1_ps(4.9f) };
        const __m512 one { _mm512_set1_ps(1.0f) };
        const __m512 negative { _mm512_set1_ps(-0.0f) };

        // Constant values for approximation
        const __m512 v28 { _mm512_set1_ps(28.0f) };
        const __m512 v378 { _mm512_set1_ps(378.0f) };
        const __m512 v3150 { _mm512_set1_ps(3150.0f) };
        const __m512 v17325 { _mm512_set1_ps(17325.0f) };
        const __m512 v62370 { _mm512_set1_ps(62370.0f) };
        const __m512 v135135 { _mm512_set1_ps(135135.0f) };

        for (std::size_t i = 0; i < END; i += SIMD_WIDTH) {
                const __m512 x { _mm512_loadu_ps(&vec[i]) };
                const __m512 x2 { _mm512_mul_ps(x, x) };

                // Check if |x| >= 6.0
                const __m512 absoluteX { _mm512_abs_ps(x)};
                const __mmask16 mask { _mm512_cmp_ps_mask(absoluteX, threshold, _CMP_GE_OQ) };
                const __m512 signs { _mm512_and_ps(x, negative) };
                const __m512 signed_one { _mm512_or_ps(signs, one) };

                // Numerator: x * (135135 + x2 * (17325 + x2 * (378 + x2)))
                __m512 numerator { _mm512_add_ps(x2, v378) };
                numerator = _mm512_fmadd_ps(x2, numerator, v17325);
                numerator = _mm512_fmadd_ps(x2, numerator, v135135);
                numerator = _mm512_mul_ps(x, numerator);

                // Denominator: 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
                __m512 denominator { _mm512_fmadd_ps(x2, v28, v3150) };
                denominator = _mm512_fmadd_ps(x2, denominator, v62370);
                denominator = _mm512_fmadd_ps(x2, denominator, v135135);

                __m512 tanh { _mm512_div_ps(numerator, denominator) };
                tanh = _mm512_mask_blend_ps(mask, tanh, signed_one);

                _mm512_storeu_ps(&result[i], tanh);
        }

        for (std::size_t i { END }; i < vec.size(); i++)
                result[i] = Fast::tanh(vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Fast::tanhDerivative(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = Fast::tanhDerivative(vec[i]);

        return result;
#else
        const uint32_t END { (vec.size() / SIMD_WIDTH) * SIMD_WIDTH };

        const Vector tanh { Fast::tanh(vec) };
        Vector result(vec.size());

        // Constant values for mask
        const __m512 threshold { _mm512_set1_ps(4.9f) };
        const __m512 zero { _mm512_setzero_ps() };
        const __m512 one { _mm512_set1_ps(1.0f) };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&vec[i]) };
                const __m512 tanhValues { _mm512_loadu_ps(&tanh[i]) };
                __m512 tanhDerivative { _mm512_fnmadd_ps(tanhValues, tanhValues, one) };

                // Check if |x| > 4.9
                const __mmask16 mask_large { _mm512_cmp_ps_mask(_mm512_abs_ps(values), threshold, _CMP_GT_OQ) };
                tanhDerivative = _mm512_mask_blend_ps(mask_large, tanhDerivative, zero);

                // Check if x == 0
                const __mmask16 mask_zero { _mm512_cmp_ps_mask(values, zero, _CMP_EQ_OQ) };
                tanhDerivative = _mm512_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm512_storeu_ps(&result[i], tanhDerivative);
        }

        for (uint32_t i { END }; i < vec.size(); i++)
                result[i] = Fast::tanhDerivative(vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}