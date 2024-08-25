#include "Matrix.h"

#include <cstring>

Matrix::Matrix(
        uint32_t width,
        uint32_t height
)  :
        m_width(width),
        m_height(height),
        m_size(width * height),
        m_data(new float[m_size](/* 0 initialized */))
{}

Matrix::Matrix(const Matrix &other) :
        m_width(other.m_width),
        m_height(other.m_height),
        m_size(other.m_size),
        m_data(new float[m_width * m_height](/* 0 initialized */))
{
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));
}

Matrix::Matrix(Matrix &&other) noexcept  :
        m_width(other.m_width),
        m_height(other.m_height),
        m_size(other.m_size),
        m_data(other.m_data)
{
        other.m_data = nullptr;
}

Matrix::~Matrix()
{
        delete [] m_data;
}

Matrix &Matrix::operator= (
        const Matrix &other
) noexcept {
        if (this == &other)
                return *this;

        if (m_size != other.m_size) {
                delete[] m_data;

                m_size = other.m_size;
                m_width = other.m_width;
                m_height = other.m_height;
                m_data = new float[m_size];
        }

        std::memcpy(m_data, other.m_data, m_size * sizeof(float));

        return *this;
}

Matrix &Matrix::operator= (
        Matrix &&other
) noexcept {
        if (this == &other)
                return *this;

        delete[] m_data;

        m_data = other.m_data;
        m_size = other.m_size;
        m_width = other.m_width;
        m_height = other.m_height;

        other.m_size = 0;
        other.m_width = 0;
        other.m_height = 0;
        other.m_data = nullptr;

        return *this;
}

Matrix Matrix::operator* (
        float scalar
) const noexcept {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Matrix result(m_width, m_height);

        for (uint32_t i { 0 }; i < m_size; i++)
                result.m_data[i] = m_data[i] * scalar;

        return result;
#else
        const uint32_t END { (m_width * m_height / SIMD_WIDTH) * SIMD_WIDTH };

        Matrix result(m_width, m_height);

        const __m512 scalarMatrix { _mm512_set1_ps(scalar) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 mulResult { _mm512_mul_ps(values, scalarMatrix) };

                _mm512_storeu_ps(&result.m_data[i], mulResult);
        }

        for (uint32_t i { END }; i < m_width * m_height; i++)
                result.m_data[i] = m_data[i] * scalar;

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Matrix::operator+= (
        const Matrix &other
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error("Matrix dimensions must match for addition.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] += other.m_data[i];
#else
        const uint32_t END { (m_width * m_height / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other.m_data[i]) };
                const __m512 addResult { _mm512_add_ps(values, otherValues) };

                _mm512_storeu_ps(&m_data[i], addResult);
        }

        for (uint32_t i { END }; i < m_width * m_height; i++)
                m_data[i] += other.m_data[i];
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Matrix::operator-= (
        const Matrix &other
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error("Matrix dimensions must match for addition.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] -= other.m_data[i];
#else
        const uint32_t END { (m_width * m_height / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other.m_data[i]) };
                const __m512 addResult { _mm512_sub_ps(values, otherValues) };

                _mm512_storeu_ps(&m_data[i], addResult);
        }

        for (uint32_t i { END }; i < m_width * m_height; i++)
                m_data[i] -= other.m_data[i];
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Matrix::operator* (
        const Vector &vec
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (vec.m_size != m_width)
                throw Logger::fatal_error("Matrix width and vector size must match.");

        Vector result(m_height);
        for (uint32_t i { 0 }; i < m_height; i++)
                for (uint32_t j { 0 }; j < m_width; j++)
                        result.m_data[i] += m_data[i * m_width + j] * vec.at(j);

        return result;
#else
        const uint32_t END { (m_width / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_height);
        for (uint32_t i { 0 }; i < m_height; i++) {
                __m512 sum { _mm512_setzero_ps() };

                for (uint32_t j { 0 }; j < END; j+=SIMD_WIDTH) {
                        const __m512 values { _mm512_loadu_ps(&m_data[i * m_width + j]) };
                        const __m512 vectorValues { _mm512_loadu_ps(&vec.m_data[j]) };
                        const __m512 product { _mm512_mul_ps(values, vectorValues) };

                        sum = _mm512_add_ps(sum, product);
                }

                result.m_data[i] = _mm512_reduce_add_ps(sum);

                for (uint32_t j { END }; j < m_width; j++)
                        result.m_data[i] += m_data[i * m_width * j] * vec.at(j);
        }

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}
