#include "Vector.h"

#include <immintrin.h>
#include <cstring>

#include "../utils/Logger.h"

Vector::Vector(
        uint32_t size
) :
        m_size(size),
        m_data(new float[m_size](/* 0 initialized */))
{}

Vector::Vector(const Vector &other) :
        m_size(other.m_size),
        m_data(new float[m_size])
{
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));
}

Vector::Vector(Vector &&other) noexcept :
        m_size(other.m_size),
        m_data(other.m_data)
{
        other.m_size = 0;
        other.m_data = nullptr;
}

Vector::~Vector()
{
        delete [] m_data;
}

Vector &Vector::operator= (
        const Vector &other
) noexcept {
        if (this == &other)
                return *this;

        if (m_size != other.m_size) {
                delete[] m_data;
                m_size = other.m_size;
                m_data = new float[m_size];
        }

        std::memcpy(m_data, other.m_data, m_size * sizeof(float));

        return *this;
}

Vector &Vector::operator= (
        Vector &&other
) noexcept {
        if (this == &other)
                return *this;

        delete[] m_data;

        m_size = other.m_size;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;

        return *this;
}

Vector Vector::operator+ (
        const Vector &other
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for addition.");

        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result.m_data[i] = m_data[i] + other[i];

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                __m512 values { _mm512_loadu_ps(&m_data[i]) };
                __m512 otherValues { _mm512_loadu_ps(&other.m_data[i]) };
                __m512 addResult { _mm512_add_ps(values, otherValues) };
                _mm512_storeu_ps(&result.m_data[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result.m_data[i] = m_data[i] + other[i];

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator* (
        float scalar
) const noexcept {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result.m_data[i] = m_data[i] * scalar;

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);

        __m512 scalarValues { _mm512_set1_ps(scalar) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                __m512 values { _mm512_loadu_ps(&m_data[i]) };
                __m512 mulResult { _mm512_mul_ps(values, scalarValues) };
                _mm512_storeu_ps(&result.m_data[i], mulResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result.m_data[i] = m_data[i] * scalar;

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator+= (
        const Vector &other
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for addition.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] += other[i];
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                __m512 values { _mm512_loadu_ps(&m_data[i]) };
                __m512 otherValues { _mm512_loadu_ps(&other.m_data[i]) };
                __m512 addResult { _mm512_add_ps(values, otherValues) };
                _mm512_storeu_ps(&m_data[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] += other[i];
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator*= (
        const Vector &other
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for multiplication.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] *= other[i];
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                __m512 values { _mm512_loadu_ps(&m_data[i]) };
                __m512 otherValues { _mm512_loadu_ps(&other.m_data[i]) };
                __m512 mulResult { _mm512_mul_ps(values, otherValues) };
                _mm512_storeu_ps(&m_data[i], mulResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] *= other[i];
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator/= (
        float scalar
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (scalar == 0.0f)
                throw Logger::fatal_error("Cannot divide by 0.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] /= scalar;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        __m512 scalarValues { _mm512_set1_ps(scalar) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                __m512 values { _mm512_loadu_ps(&m_data[i]) };
                __m512 divResult { _mm512_div_ps(values, scalarValues) };
                _mm512_storeu_ps(&m_data[i], divResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] /= scalar;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}