#include "../math/FastMath.h"

#include <algorithm>
#include <cmath>

// Internal single float functions
namespace Fast {

        inline float tanh(float x)
        {
                // Around 6-8 times faster than std::tanh in release mode

                // For |x| > 4.9, tanh(x) is approximately 1 or -1. Returns at 4.9
                // since it's the highest decimal the output isn't higher than 1
                if (std::abs(x) >= 4.9f)
                        return std::copysign(1.0f, x);

                // 7th-order approximation (accurate within 0.000085).
                // https://www.desmos.com/calculator/2myik1oe4x
                const float x2 { x * x };
                return x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) / (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }
        inline float tanhDerivative(float x)
        {
                // Uses the tanhFast to calculate the derivative way faster than
                // with using std::tanh, also more accurate than tanhFast.

                // tanh'(0) = 1
                if (x == 0.0f)
                        return 1.0f;

                // After this point, the approximation of the derivative goes negative.
                // Instead, the tanh' goes closer and closer to 0 (accurate within 0.000170).
                // Returns at 4.9 since it's the highest decimal the value is positive.
                if (std::abs(x) > 4.9f)
                        return 0.0f;

                const float tanh { Fast::tanh(x) };
        #ifdef DEBUG_MODE_ENABLED
                const float value { 1.0f - tanh * tanh };
                if (value < 0.0f)
                        throw Logger::fatal_error("Returned value from tanhDerivativeFast is negative.");

                return value;
        #else
                return 1.0f - tanh * tanh;
        #endif // DEBUG_MODE_ENABLED
        }

} // namespace Fast

Vector Fast::relu(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = std::max(0.0f, vec.at(i));

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
                result[i] = std::max(0.0f, vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Fast::reluDerivative(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = vec.at(i) >= 0.0f ? 1.0f : 0.0f;

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
                result[i] = vec.at(i) >= 0.0f ? 1.0f : 0.0f;

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Fast::tanh(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (std::size_t i { 0 }; i < vec.size(); i++)
                result[i] = Fast::tanh(vec.at(i));

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

        for (uint32_t i { 0 }; i < END; i += SIMD_WIDTH) {
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

        for (uint32_t i { END }; i < vec.size(); i++)
                result[i] = ::Fast::tanh(vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Fast::tanhDerivative(const Vector &vec)
{
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = Fast::tanhDerivative(vec.at(i));

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
                result[i] = ::Fast::tanhDerivative(vec.at(i));

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}