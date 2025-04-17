#include "Math.h"

#include <immintrin.h>

#include "Base.h"

// An equivalent of _mm512_abs_ps for SSE does not exists, so this just
// changes the most significant bit of the numbers to 0 (positive).
#define _mm_abs_ps(X) _mm_and_ps(X, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)))

// An equivalent of _mm512_reduce_add_ps for SSE does not exists.
static inline float reduce_add_ps_sse(__m128 v) {
        __m128 shuf { _mm_movehdup_ps(v) };
        __m128 sums { _mm_add_ps(v, shuf) };
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

#define _mm_reduce_add_ps(X) reduce_add_ps_sse(X)

template<> void Math<MATH_SSE>::sum(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 otherValues { _mm_loadu_ps(&second[i]) };
                const __m128 sumResult { _mm_add_ps(values, otherValues) };

                _mm_storeu_ps(&result[i], sumResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] + second[i];
}

template<> void Math<MATH_SSE>::sub(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 otherValues { _mm_loadu_ps(&second[i]) };
                const __m128 subResult { _mm_sub_ps(values, otherValues) };

                _mm_storeu_ps(&result[i], subResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] - second[i];
}

template<> void Math<MATH_SSE>::mul(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 otherValues { _mm_loadu_ps(&second[i]) };
                const __m128 mulResult { _mm_mul_ps(values, otherValues) };

                _mm_storeu_ps(&result[i], mulResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] * second[i];
}

template<> void Math<MATH_SSE>::div(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 otherValues { _mm_loadu_ps(&second[i]) };
                const __m128 divResult { _mm_div_ps(values, otherValues) };

                _mm_storeu_ps(&result[i], divResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] / second[i];
}

template<> void Math<MATH_SSE>::sum(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };
        const __m128 scalarValues { _mm_set1_ps(scalar) };
        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 sumResult { _mm_add_ps(values, scalarValues) };

                _mm_storeu_ps(&result[i], sumResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] + scalar;
}

template<> void Math<MATH_SSE>::sub(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };
        const __m128 scalarValues { _mm_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 subResult { _mm_sub_ps(values, scalarValues) };

                _mm_storeu_ps(&result[i], subResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] - scalar;
}

template<> void Math<MATH_SSE>::mul(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };
        const __m128 scalarValues { _mm_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 mulResult { _mm_mul_ps(values, scalarValues) };

                _mm_storeu_ps(&result[i], mulResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] * scalar;
}

template<> void Math<MATH_SSE>::div(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };
        const __m128 scalarValues { _mm_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 divResult { _mm_div_ps(values, scalarValues) };

                _mm_storeu_ps(&result[i], divResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = first[i] / scalar;
}

template<> void Math<MATH_SSE>::tanh(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        // Constant values for mask
        const __m128 threshold { _mm_set1_ps(4.9f) };
        const __m128 one { _mm_set1_ps(1.0f) };
        const __m128 negative { _mm_set1_ps(-0.0f) };

        // Constant values for approximation
        const __m128 v28 { _mm_set1_ps(28.0f) };
        const __m128 v378 { _mm_set1_ps(378.0f) };
        const __m128 v3150 { _mm_set1_ps(3150.0f) };
        const __m128 v17325 { _mm_set1_ps(17325.0f) };
        const __m128 v62370 { _mm_set1_ps(62370.0f) };
        const __m128 v135135 { _mm_set1_ps(135135.0f) };

        for (uint32_t i { 0 }; i < END; i += SSE_SIMD_WIDTH) {
                const __m128 x { _mm_loadu_ps(&data[i]) };
                const __m128 x2 { _mm_mul_ps(x, x) };

                // Check if |x| >= 6.0
                const __m128 absoluteX { _mm_abs_ps(x) };
                const __mmask8 mask { _mm_cmp_ps_mask(absoluteX, threshold, _CMP_GE_OQ) };
                const __m128 signs { _mm_and_ps(x, negative) };
                const __m128 signed_one { _mm_or_ps(signs, one) };

                // Numerator: x * (135135 + x2 * (17325 + x2 * (378 + x2)))
                __m128 numerator { _mm_add_ps(x2, v378) };
                numerator = _mm_fmadd_ps(x2, numerator, v17325);
                numerator = _mm_fmadd_ps(x2, numerator, v135135);
                numerator = _mm_mul_ps(x, numerator);

                // Denominator: 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
                __m128 denominator { _mm_fmadd_ps(x2, v28, v3150) };
                denominator = _mm_fmadd_ps(x2, denominator, v62370);
                denominator = _mm_fmadd_ps(x2, denominator, v135135);

                __m128 tanh { _mm_div_ps(numerator, denominator) };
                tanh = _mm_mask_blend_ps(mask, tanh, signed_one);

                _mm_storeu_ps(&result[i], tanh);
        }

        for (uint32_t i { END }; i < size; i++) {
                const float x { data[i] };
                if (std::abs(x) >= 4.9f) {
                        result[i] = std::copysign(1.0f, x);
                        continue;
                }
                const float x2 { x * x };
                result[i] = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) /
                            (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }
}

template<> void Math<MATH_SSE>::tanhDerivative(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        Math<MATH_SSE>::tanh(size, data, result);

        // Constant values for mask
        const __m128 threshold { _mm_set1_ps(4.9f) };
        const __m128 zero { _mm_setzero_ps() };
        const __m128 one { _mm_set1_ps(1.0f) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&data[i]) };
                // Tanh values are stored in the result array, then overwritten.
                const __m128 tanhValues { _mm_loadu_ps(&result[i]) };
                __m128 tanhDerivative { _mm_fnmadd_ps(tanhValues, tanhValues, one) };

                // Check if |x| > 4.9
                const __mmask8 mask_large { _mm_cmp_ps_mask(_mm_abs_ps(values), threshold, _CMP_GT_OQ) };
                tanhDerivative = _mm_mask_blend_ps(mask_large, tanhDerivative, zero);

                // Check if x == 0
                const __mmask16 mask_zero { _mm_cmp_ps_mask(values, zero, _CMP_EQ_OQ) };
                tanhDerivative = _mm_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm_storeu_ps(&result[i], tanhDerivative);
        }

        for (uint32_t i { END }; i < size; i++) {
                const float x { data[i] };
                if (x == 0.0f) {
                        result[i] = 1.0f;
                        continue;
                }
                if (std::abs(x) > 4.9f) {
                        result[i] = 0.0f;
                        continue;
                }
                const float tanh { result[i] };
                result[i] = 1 - tanh * tanh;
        }
}

template<> void Math<MATH_SSE>::ReLU(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        const __m128 zero { _mm_setzero_ps() };
        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 x { _mm_loadu_ps(&data[i]) };
                const __m128 relu { _mm_max_ps(zero, x) };

                _mm_storeu_ps(&result[i], relu);
        }

        for (uint32_t i { END }; i < size; i++)
                result[i] = std::max(0.0f, data[i]);
}

template<> void Math<MATH_SSE>::ReLUDerivative(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        const __m128 zero { _mm_setzero_ps() };
        const __m128 one { _mm_set1_ps(1.0f) };
        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 x { _mm_loadu_ps(&data[i]) };
                const __mmask8 mask { _mm_cmp_ps_mask(x, zero, _CMP_GT_OQ) }; // Compare x > 0
                const __m128 reluDerivative { _mm_mask_blend_ps(mask, zero, one) };

                _mm_storeu_ps(&result[i], reluDerivative);
        }

        for (uint32_t i { END }; i < size; i++)
                result[i] = data[i] >= 0.0f ? 1.0f : 0.0f;
}

template<> void Math<MATH_SSE>::min(uint32_t size, const float data[], float min, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        const __m128 minValues { _mm_set1_ps(min) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&data[i]) };
                const __m128 minResult { _mm_min_ps(values, minValues) };

                _mm_storeu_ps(&result[i], minResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = std::min(data[i], min);
}

template<> void Math<MATH_SSE>::max(uint32_t size, const float data[], float max, float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        const __m128 maxValues { _mm_set1_ps(max) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&data[i]) };
                const __m128 maxResult { _mm_max_ps(values, maxValues) };

                _mm_storeu_ps(&result[i], maxResult);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = std::max(data[i], max);
}

template<> void Math<MATH_SSE>::clamp(uint32_t size, const float data[], float min, float max, float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        const __m128 minValues { _mm_set1_ps(min) };
        const __m128 maxValues { _mm_set1_ps(max) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&data[i]) };
                const __m128 clamp { _mm_max_ps(_mm_min_ps(maxValues, values), minValues) };

                _mm_storeu_ps(&result[i], clamp);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = std::clamp(data[i], min, max);
}

template<> void Math<MATH_SSE>::min(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 otherValues { _mm_loadu_ps(&second[i]) };
                const __m128 minValues { _mm_min_ps(otherValues, values) };

                _mm_storeu_ps(&result[i], minValues);
        }

        for (uint32_t i { END }; i < size; i++)
                result[i] = std::min(first[i], second[i]);
}

template<> void Math<MATH_SSE>::max(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&first[i]) };
                const __m128 otherValues { _mm_loadu_ps(&second[i]) };
                const __m128 maxValues { _mm_max_ps(otherValues, values) };

                _mm_storeu_ps(&result[i], maxValues);
        }

        for (uint32_t i { END }; i < size; i++)
                result[i] = std::max(first[i], second[i]);
}

template<> void Math<MATH_SSE>::clamp(uint32_t size, const float data[], const float min[], const float max[], float result[])
{
        const uint32_t END { size & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=SSE_SIMD_WIDTH) {
                const __m128 values { _mm_loadu_ps(&data[i]) };
                const __m128 minValues { _mm_loadu_ps(&min[i]) };
                const __m128 maxValues { _mm_loadu_ps(&max[i]) };
                const __m128 clamp { _mm_max_ps(_mm_min_ps(maxValues, values), minValues) };

                _mm_storeu_ps(&result[i], clamp);
        }

        for (uint32_t i { END }; i < size; ++i)
                result[i] = std::clamp(data[i], min[i], max[i]);
}

template<> void Math<MATH_SSE>::matvec_mul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[])
{
        const uint32_t END { width & ~(SSE_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < height; i++) {
                __m128 sum { _mm_setzero_ps() };

                for (uint32_t j { 0 }; j < END; j+=SSE_SIMD_WIDTH) {
                        const __m128 values { _mm_loadu_ps(&matrix[i * width + j]) };
                        const __m128 vectorValues { _mm_loadu_ps(&vector[j]) };
                        const __m128 product { _mm_mul_ps(values, vectorValues) };

                        sum = _mm_add_ps(sum, product);
                }

                result[i] = _mm_reduce_add_ps(sum);

                for (uint32_t j { END }; j < width; j++)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}

// An equivalent of _mm512_abs_ps for AVX does not exists, so this just
// changes the most significant bit of the numbers to 0 (positive).
#define _mm256_abs_ps(X) _mm256_and_ps(X, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)))

// An equivalent of _mm512_reduce_add_ps for SSE does not exists.
static inline float reduce_add_ps_avx(__m256 v) {
        __m128 low = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
}

#define _mm256_reduce_add_ps(X) reduce_add_ps_avx(X)

template<> void Math<MATH_AVX>::sum(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 otherValues { _mm256_loadu_ps(&second[i]) };
                const __m256 sumResult { _mm256_add_ps(values, otherValues) };

                _mm256_storeu_ps(&result[i], sumResult);
        }

        Math<MATH_SSE>::sum(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX>::sub(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 otherValues { _mm256_loadu_ps(&second[i]) };
                const __m256 subResult { _mm256_sub_ps(values, otherValues) };

                _mm256_storeu_ps(&result[i], subResult);
        }

        Math<MATH_SSE>::sub(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX>::mul(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 otherValues { _mm256_loadu_ps(&second[i]) };
                const __m256 mulResult { _mm256_mul_ps(values, otherValues) };

                _mm256_storeu_ps(&result[i], mulResult);
        }

        Math<MATH_SSE>::mul(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX>::div(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 otherValues { _mm256_loadu_ps(&second[i]) };
                const __m256 divResult { _mm256_div_ps(values, otherValues) };

                _mm256_storeu_ps(&result[i], divResult);
        }

        Math<MATH_SSE>::div(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX>::sum(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };
        const __m256 scalarValues { _mm256_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 sumResult { _mm256_add_ps(values, scalarValues) };

                _mm256_storeu_ps(&result[i], sumResult);
        }

        Math<MATH_SSE>::sum(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX>::sub(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };
        const __m256 scalarValues { _mm256_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 subResult { _mm256_sub_ps(values, scalarValues) };

                _mm256_storeu_ps(&result[i], subResult);
        }

        Math<MATH_SSE>::sub(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX>::mul(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };
        const __m256 scalarValues { _mm256_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 mulResult { _mm256_mul_ps(values, scalarValues) };

                _mm256_storeu_ps(&result[i], mulResult);
        }

        Math<MATH_SSE>::mul(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX>::div(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };
        const __m256 scalarValues { _mm256_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 divResult { _mm256_div_ps(values, scalarValues) };

                _mm256_storeu_ps(&result[i], divResult);
        }

        Math<MATH_SSE>::div(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX>::tanh(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        // Constant values for mask
        const __m256 threshold { _mm256_set1_ps(4.9f) };
        const __m256 one { _mm256_set1_ps(1.0f) };
        const __m256 negative { _mm256_set1_ps(-0.0f) };

        // Constant values for approximation
        const __m256 v28 { _mm256_set1_ps(28.0f) };
        const __m256 v378 { _mm256_set1_ps(378.0f) };
        const __m256 v3150 { _mm256_set1_ps(3150.0f) };
        const __m256 v17325 { _mm256_set1_ps(17325.0f) };
        const __m256 v62370 { _mm256_set1_ps(62370.0f) };
        const __m256 v135135 { _mm256_set1_ps(135135.0f) };

        for (uint32_t i { 0 }; i < END; i += AVX_SIMD_WIDTH) {
                const __m256 x { _mm256_loadu_ps(&data[i]) };
                const __m256 x2 { _mm256_mul_ps(x, x) };

                // Check if |x| >= 6.0
                const __m256 absoluteX { _mm256_abs_ps(x) };
                const __mmask8 mask { _mm256_cmp_ps_mask(absoluteX, threshold, _CMP_GE_OQ) };
                const __m256 signs { _mm256_and_ps(x, negative) };
                const __m256 signed_one { _mm256_or_ps(signs, one) };

                // Numerator: x * (135135 + x2 * (17325 + x2 * (378 + x2)))
                __m256 numerator { _mm256_add_ps(x2, v378) };
                numerator = _mm256_fmadd_ps(x2, numerator, v17325);
                numerator = _mm256_fmadd_ps(x2, numerator, v135135);
                numerator = _mm256_mul_ps(x, numerator);

                // Denominator: 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
                __m256 denominator { _mm256_fmadd_ps(x2, v28, v3150) };
                denominator = _mm256_fmadd_ps(x2, denominator, v62370);
                denominator = _mm256_fmadd_ps(x2, denominator, v135135);

                __m256 tanh { _mm256_div_ps(numerator, denominator) };
                tanh = _mm256_mask_blend_ps(mask, tanh, signed_one);

                _mm256_storeu_ps(&result[i], tanh);
        }

        // Trying to use SSE for the remaining part of the array.
        Math<MATH_SSE>::tanh(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX>::tanhDerivative(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        Math<MATH_AVX>::tanh(size, data, result);

        // Constant values for mask
        const __m256 threshold { _mm256_set1_ps(4.9f) };
        const __m256 zero { _mm256_setzero_ps() };
        const __m256 one { _mm256_set1_ps(1.0f) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&data[i]) };
                // Tanh values are stored in the result array, then overwritten.
                const __m256 tanhValues { _mm256_loadu_ps(&result[i]) };
                __m256 tanhDerivative { _mm256_fnmadd_ps(tanhValues, tanhValues, one) };

                // Check if |x| > 4.9
                const __mmask8 mask_large { _mm256_cmp_ps_mask(_mm256_abs_ps(values), threshold, _CMP_GT_OQ) };
                tanhDerivative = _mm256_mask_blend_ps(mask_large, tanhDerivative, zero);

                // Check if x == 0
                const __mmask16 mask_zero { _mm256_cmp_ps_mask(values, zero, _CMP_EQ_OQ) };
                tanhDerivative = _mm256_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm256_storeu_ps(&result[i], tanhDerivative);
        }

        // Trying to use SSE for the remaining part of the array.
        Math<MATH_SSE>::tanhDerivative(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX>::ReLU(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        const __m256 zero { _mm256_setzero_ps() };
        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 x { _mm256_loadu_ps(&data[i]) };
                const __m256 relu { _mm256_max_ps(zero, x) };

                _mm256_storeu_ps(&result[i], relu);
        }

        // Trying to use SSE for the remaining part of the array.
        Math<MATH_SSE>::ReLU(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX>::ReLUDerivative(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        const __m256 zero { _mm256_setzero_ps() };
        const __m256 one { _mm256_set1_ps(1.0f) };
        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 x { _mm256_loadu_ps(&data[i]) };
                const __mmask8 mask { _mm256_cmp_ps_mask(x, zero, _CMP_GT_OQ) }; // Compare x > 0
                const __m256 reluDerivative { _mm256_mask_blend_ps(mask, zero, one) };

                _mm256_storeu_ps(&result[i], reluDerivative);
        }

        // Trying to use SSE for the remaining part of the array.
        Math<MATH_SSE>::ReLUDerivative(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX>::min(uint32_t size, const float data[], float min, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        const __m256 minValues { _mm256_set1_ps(min) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&data[i]) };
                const __m256 minResult { _mm256_min_ps(values, minValues) };

                _mm256_storeu_ps(&result[i], minResult);
        }

        Math<MATH_SSE>::min(size - END, data + END, min, result + END);
}


template<> void Math<MATH_AVX>::max(uint32_t size, const float data[], float max, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        const __m256 maxValues { _mm256_set1_ps(max) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&data[i]) };
                const __m256 maxResult { _mm256_max_ps(values, maxValues) };

                _mm256_storeu_ps(&result[i], maxResult);
        }

        Math<MATH_SSE>::max(size - END, data + END, max, result + END);
}

template<> void Math<MATH_AVX>::clamp(uint32_t size, const float data[], float min, float max, float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        const __m256 minValues { _mm256_set1_ps(min) };
        const __m256 maxValues { _mm256_set1_ps(max) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&data[i]) };
                const __m256 clamp { _mm256_max_ps(_mm256_min_ps(maxValues, values), minValues) };

                _mm256_storeu_ps(&result[i], clamp);
        }

        Math<MATH_SSE>::clamp(size - END, data + END, min, max, result + END);
}

template<> void Math<MATH_AVX>::min(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 otherValues { _mm256_loadu_ps(&second[i]) };
                const __m256 minValues { _mm256_min_ps(otherValues, values) };

                _mm256_storeu_ps(&result[i], minValues);
        }

        Math<MATH_SSE>::min(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX>::max(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&first[i]) };
                const __m256 otherValues { _mm256_loadu_ps(&second[i]) };
                const __m256 maxValues { _mm256_max_ps(otherValues, values) };

                _mm256_storeu_ps(&result[i], maxValues);
        }

        Math<MATH_SSE>::max(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX>::clamp(uint32_t size, const float data[], const float min[], const float max[], float result[])
{
        const uint32_t END { size & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX_SIMD_WIDTH) {
                const __m256 values { _mm256_loadu_ps(&data[i]) };
                const __m256 minValues { _mm256_loadu_ps(&min[i]) };
                const __m256 maxValues { _mm256_loadu_ps(&max[i]) };
                const __m256 clamp { _mm256_max_ps(_mm256_min_ps(maxValues, values), minValues) };

                _mm256_storeu_ps(&result[i], clamp);
        }

        Math<MATH_SSE>::clamp(size - END, data + END, min + END, max + END, result + END);
}

template<> void Math<MATH_AVX>::matvec_mul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[])
{
        const uint32_t END { width & ~(AVX_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < height; i++) {
                __m256 sum { _mm256_setzero_ps() };

                for (uint32_t j { 0 }; j < END; j+=AVX_SIMD_WIDTH) {
                        const __m256 values { _mm256_loadu_ps(&matrix[i * width + j]) };
                        const __m256 vectorValues { _mm256_loadu_ps(&vector[j]) };
                        const __m256 product { _mm256_mul_ps(values, vectorValues) };

                        sum = _mm256_add_ps(sum, product);
                }

                result[i] = _mm256_reduce_add_ps(sum);

                for (uint32_t j { END }; j < width; j++)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}

template<> void Math<MATH_AVX512>::sum(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&second[i]) };
                const __m512 sumResult { _mm512_add_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], sumResult);
        }

        Math<MATH_AVX>::sum(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX512>::sub(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&second[i]) };
                const __m512 subResult { _mm512_sub_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], subResult);
        }

        Math<MATH_AVX>::sub(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX512>::mul(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&second[i]) };
                const __m512 mulResult { _mm512_mul_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], mulResult);
        }

        Math<MATH_AVX>::mul(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX512>::div(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&second[i]) };
                const __m512 divResult { _mm512_div_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], divResult);
        }

        Math<MATH_AVX>::div(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX512>::sum(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };
        const __m512 scalarValues { _mm512_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 sumResult { _mm512_add_ps(values, scalarValues) };

                _mm512_storeu_ps(&result[i], sumResult);
        }

        Math<MATH_AVX>::sum(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX512>::sub(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };
        const __m512 scalarValues { _mm512_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 subResult { _mm512_sub_ps(values, scalarValues) };

                _mm512_storeu_ps(&result[i], subResult);
        }

        Math<MATH_AVX>::sub(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX512>::mul(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };
        const __m512 scalarValues { _mm512_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 mulResult { _mm512_mul_ps(values, scalarValues) };

                _mm512_storeu_ps(&result[i], mulResult);
        }

        Math<MATH_AVX>::mul(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX512>::div(uint32_t size, const float first[], float scalar, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };
        const __m512 scalarValues { _mm512_set1_ps(scalar) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 divResult { _mm512_div_ps(values, scalarValues) };

                _mm512_storeu_ps(&result[i], divResult);
        }

        Math<MATH_AVX>::div(size - END, first + END, scalar, result + END);
}

template<> void Math<MATH_AVX512>::tanh(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

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

        for (uint32_t i { 0 }; i < END; i += AVX512_SIMD_WIDTH) {
                const __m512 x { _mm512_loadu_ps(&data[i]) };
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

        // Trying to use AVX for the remaining part of the array.
        Math<MATH_AVX>::tanh(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX512>::tanhDerivative(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        Math<MATH_AVX512>::tanh(size, data, result);

        // Constant values for mask
        const __m512 threshold { _mm512_set1_ps(4.9f) };
        const __m512 zero { _mm512_setzero_ps() };
        const __m512 one { _mm512_set1_ps(1.0f) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&data[i]) };
                // Tanh values are stored in the result array, then overwritten.
                const __m512 tanhValues { _mm512_loadu_ps(&result[i]) };
                __m512 tanhDerivative { _mm512_fnmadd_ps(tanhValues, tanhValues, one) };

                // Check if |x| > 4.9
                const __mmask16 mask_large { _mm512_cmp_ps_mask(_mm512_abs_ps(values), threshold, _CMP_GT_OQ) };
                tanhDerivative = _mm512_mask_blend_ps(mask_large, tanhDerivative, zero);

                // Check if x == 0
                const __mmask16 mask_zero { _mm512_cmp_ps_mask(values, zero, _CMP_EQ_OQ) };
                tanhDerivative = _mm512_mask_blend_ps(mask_zero, tanhDerivative, one);

                _mm512_storeu_ps(&result[i], tanhDerivative);
        }

        // Trying to use AVX for the remaining part of the array.
        Math<MATH_AVX>::tanhDerivative(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX512>::ReLU(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        const __m512 zero { _mm512_setzero_ps() };
        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 x { _mm512_loadu_ps(&data[i]) };
                const __m512 relu { _mm512_max_ps(zero, x) };

                _mm512_storeu_ps(&result[i], relu);
        }

        // Trying to use AVX for the remaining part of the array.
        Math<MATH_AVX>::ReLU(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX512>::ReLUDerivative(uint32_t size, const float data[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        const __m512 zero { _mm512_setzero_ps() };
        const __m512 one { _mm512_set1_ps(1.0f) };
        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 x { _mm512_loadu_ps(&data[i]) };
                const __mmask16 mask { _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) }; // Compare x > 0
                const __m512 reluDerivative { _mm512_mask_blend_ps(mask, zero, one) };

                _mm512_storeu_ps(&result[i], reluDerivative);
        }

        // Trying to use AVX for the remaining part of the array.
        Math<MATH_AVX>::ReLUDerivative(size - END, data + END, result + END);
}

template<> void Math<MATH_AVX512>::min(uint32_t size, const float data[], float min, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        const __m512 minValues { _mm512_set1_ps(min) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&data[i]) };
                const __m512 minResult { _mm512_min_ps(values, minValues) };

                _mm512_storeu_ps(&result[i], minResult);
        }

        Math<MATH_AVX>::min(size - END, data + END, min, result + END);
}

template<> void Math<MATH_AVX512>::max(uint32_t size, const float data[], float max, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        const __m512 maxValues { _mm512_set1_ps(max) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&data[i]) };
                const __m512 maxResult { _mm512_max_ps(values, maxValues) };

                _mm512_storeu_ps(&result[i], maxResult);
        }

        Math<MATH_AVX>::max(size - END, data + END, max, result + END);
}

template<> void Math<MATH_AVX512>::clamp(uint32_t size, const float data[], float min, float max, float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        const __m512 minValues { _mm512_set1_ps(min) };
        const __m512 maxValues { _mm512_set1_ps(max) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&data[i]) };
                const __m512 clamp { _mm512_max_ps(_mm512_min_ps(maxValues, values), minValues) };

                _mm512_storeu_ps(&result[i], clamp);
        }

        Math<MATH_AVX>::clamp(size - END, data + END, min, max, result + END);
}

template<> void Math<MATH_AVX512>::min(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&second[i]) };
                const __m512 minValues { _mm512_min_ps(otherValues, values) };

                _mm512_storeu_ps(&result[i], minValues);
        }

        Math<MATH_AVX>::min(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX512>::max(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&first[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&second[i]) };
                const __m512 maxValues { _mm512_max_ps(otherValues, values) };

                _mm512_storeu_ps(&result[i], maxValues);
        }

        Math<MATH_AVX>::max(size - END, first + END, second + END, result + END);
}

template<> void Math<MATH_AVX512>::clamp(uint32_t size, const float data[], const float min[], const float max[], float result[])
{
        const uint32_t END { size & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < END; i+=AVX512_SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&data[i]) };
                const __m512 minValues { _mm512_loadu_ps(&min[i]) };
                const __m512 maxValues { _mm512_loadu_ps(&max[i]) };
                const __m512 clamp { _mm512_max_ps(_mm512_min_ps(maxValues, values), minValues) };

                _mm512_storeu_ps(&result[i], clamp);
        }

        Math<MATH_AVX>::clamp(size - END, data + END, min + END, max + END, result + END);
}

template<> void Math<MATH_AVX512>::matvec_mul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[])
{
        const uint32_t END { width & ~(AVX512_SIMD_WIDTH - 1) };

        for (uint32_t i { 0 }; i < height; i++) {
                __m512 sum { _mm512_setzero_ps() };

                for (uint32_t j { 0 }; j < END; j+=AVX512_SIMD_WIDTH) {
                        const __m512 values { _mm512_loadu_ps(&matrix[i * width + j]) };
                        const __m512 vectorValues { _mm512_loadu_ps(&vector[j]) };
                        const __m512 product { _mm512_mul_ps(values, vectorValues) };

                        sum = _mm512_add_ps(sum, product);
                }

                result[i] = _mm512_reduce_add_ps(sum);

                for (uint32_t j { END }; j < width; j++) {
                        result[i] += matrix[i * width + j] * vector[j];
                }
        }
}