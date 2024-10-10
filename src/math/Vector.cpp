#include "Vector.h"

#include <cstring>
#include <numeric>

Vector::Vector(
        uint32_t size,
        float value
) :
        m_size(size),
        m_data(new float[m_size](value))
{}

Vector::Vector(
        const std::initializer_list<float> &values
) :
        m_size(values.size()),
        m_data(new float[m_size])
{
        std::memcpy(m_data, values.begin(), m_size * sizeof(float));
}

Vector::Vector(
        uint32_t size,
        const float *data
) :
        m_size(size),
        m_data(new float[m_size])
{
        std::memcpy(m_data, data, m_size * sizeof(float));
}

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
                result[i] = m_data[i] + other.at(i);

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 addResult { _mm512_add_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] + other.at(i);

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator- (
        const Vector &other
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for subtraction.");

        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result[i] = m_data[i] - other.at(i);

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 subResult { _mm512_sub_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], subResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] - other.at(i);

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator* (
        const Vector &other
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for multiplication.");

        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result[i] = m_data[i] * other.at(i);

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 addResult { _mm512_mul_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] * other.at(i);

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator/ (
        const Vector &other
) const {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (other.m_size != m_size)
                throw Logger::fatal_error("Vector lengths must match for division.");

        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result[i] = m_data[i] / other.at(i);

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 divResult { _mm512_div_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], divResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] / other.at(i);

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator+= (
        const Vector &other
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for addition.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] += other.at(i);
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 addResult { _mm512_add_ps(values, otherValues) };

                _mm512_storeu_ps(&m_data[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] += other.at(i);
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator-= (
        const Vector &other
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for subtraction.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] -= other.at(i);
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 addResult { _mm512_sub_ps(values, otherValues) };

                _mm512_storeu_ps(&m_data[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] -= other.at(i);
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator*= (
        const Vector &other
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector lengths must match for multiplication.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] *= other.at(i);
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __m512 mulResult { _mm512_mul_ps(values, otherValues) };

                _mm512_storeu_ps(&m_data[i], mulResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] *= other.at(i);
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

void Vector::operator/= (
        float scalar
) {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        if (scalar == 0.0f)
                throw Logger::fatal_error("Cannot divide by 0.");

        for (uint32_t i { 0 }; i < m_size; i++)
                m_data[i] /= scalar;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        const __m512 scalarValues { _mm512_set1_ps(scalar) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 divResult { _mm512_div_ps(values, scalarValues) };

                _mm512_storeu_ps(&m_data[i], divResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                m_data[i] /= scalar;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator* (
        float scalar
) const noexcept {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result[i] = m_data[i] * scalar;

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);

        const __m512 scalarValues { _mm512_set1_ps(scalar) };
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 mulResult { _mm512_mul_ps(values, scalarValues) };

                _mm512_storeu_ps(&result[i], mulResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] * scalar;

        return result;
#endif  // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator* (
        const float *array
) const noexcept {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result[i] = m_data[i] * array[i];

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&array[i]) };
                const __m512 mulResult { _mm512_mul_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], mulResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] * array[i];

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

Vector Vector::operator- (
        const float *array
) const noexcept {
#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        Vector result(m_size);
        for (uint32_t i { 0 }; i < m_size; i++)
                result[i] = m_data[i] - array[i];

        return result;
#else
        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        Vector result(m_size);
        for (uint32_t i { 0 }; i < END; i+=SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&array[i]) };
                const __m512 addResult { _mm512_sub_ps(values, otherValues) };

                _mm512_storeu_ps(&result[i], addResult);
        }

        for (uint32_t i { END }; i < m_size; i++)
                result[i] = m_data[i] - array[i];

        return result;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}

[[nodiscard]] bool Vector::operator== (
        const Vector &other
) const noexcept {
        if (m_size != other.m_size)
                return false;

#if defined DEBUG_MODE_ENABLED || defined DISABLE_AVX512
        for (uint32_t i { 0 }; i < m_size; i++)
                if (m_data[i] != other.at(i))
                        return false;

        return true;
#else

        const uint32_t END { (m_size / SIMD_WIDTH) * SIMD_WIDTH };

        for (uint32_t i = 0; i < END; i += SIMD_WIDTH) {
                const __m512 values { _mm512_loadu_ps(&m_data[i]) };
                const __m512 otherValues { _mm512_loadu_ps(&other[i]) };
                const __mmask16 cmp { _mm512_cmp_ps_mask(values, otherValues, _CMP_NEQ_OQ) };

                // Move the mask to an integer
                if (cmp != 0)
                        return false;
        }

        // Handle the remaining elements that do not fit into a full AVX register
        for (uint32_t i { END }; i < m_size; i++)
                if (m_data[i] != other.m_data[i])
                        return false;

        return true;
#endif // DEBUG_MODE_ENABLED || DISABLE_AVX512
}