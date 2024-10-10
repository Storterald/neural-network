#include "Utils.h"

#include <algorithm>

Vector Utils::min(
        const Vector &vec,
        const Vector &other
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (vec.size() != other.size())
                throw Logger::fatal_error("Vector sizes must match.");

        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = std::min(vec.at(i), other.at(i));

        return result;
#else
        const uint32_t END { (vec.size() / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(vec.size());

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&vec[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 minValues { _mm512_min_ps(otherValues, values) };

                _mm512_storeu_ps(&result[i], minValues);
        }

        for (uint32_t i { END }; i < vec.size(); i++)
                result[i] = std::min(vec.at(i), other.at(i));

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Utils::clamp(
        const Vector &vec,
        float min,
        float max
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(vec.size());
        for (uint32_t i { 0 }; i < vec.size(); i++)
                result[i] = std::clamp(vec.at(i), min, max);

        return result;
#else
        const uint32_t END { (vec.size() / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(vec.size());

        const __m512 minValues { _mm512_set1_ps(min) };
        const __m512 maxValues { _mm512_set1_ps(max) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&vec[i]) };

                const __m512 clamped { _mm512_max_ps(minValues, _mm512_min_ps(maxValues, values)) };

                _mm512_storeu_ps(&result[i], clamped);
        }

        for (uint32_t i { END }; i < vec.size(); i++)
                result[i] = std::clamp(vec.at(i), min, max);

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

float Utils::maxElement(
        const Vector &vec
) {
        float v { vec.at(0) };
        for (uint32_t i { 1 }; i < vec.size(); i++)
                if (vec.at(i) > v)
                        v = vec.at(i);

        return v;
}

uint32_t Utils::maxElementIndex(
        const Vector &vec
) {
        uint32_t index { 0 };
        for (uint32_t i { 1 }; i < vec.size(); i++)
                if (vec.at(i) > vec.at(index))
                        index = i;

        return index;
}