#include "Matrix.h"

Matrix::Matrix(
        uint32_t width,
        uint32_t height
) : Data(width * height),
        m_width(width),
        m_height(height)
{}

Matrix::Matrix(
        std::initializer_list<std::initializer_list<float>> values
) : Data(values.begin()->size() * values.size()),
        m_width(values.begin()->size()),
        m_height(values.size())
{
#ifdef DEBUG_MODE_ENABLED
        if (std::any_of(values.begin(), values.end(), [this](auto &vs) -> bool { return vs.size() != m_width; }))
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "All initializer lists passed to Matrix() must have the same size.");
#endif // DEBUG_MODE_ENABLED

        for (int i{}; i < m_height; ++i)
                std::memcpy((*this)[i], values.begin()[i].begin(), m_width * sizeof(float));
}

float *Matrix::operator[](uint32_t row)
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return m_data + row * m_width;
}

const float *Matrix::at(uint32_t row) const
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return m_data + row * m_width;
}

float &Matrix::operator[] (
        std::pair<uint32_t, uint32_t> position
) {
#ifdef DEBUG_MODE_ENABLED
        if (position.first >= m_height || position.second >= m_width)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return m_data[position.first * m_width + position.second];
}

float Matrix::at(
        uint32_t row,
        uint32_t height
) const {
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height || height >= m_width)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return m_data[row * m_width + height];
}

Matrix Matrix::operator+ (
        const Matrix &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        Matrix result(m_width, m_height);
        _MATH(sum, m_size, m_data, other.m_data, result.m_data);

        return result;
}

Matrix Matrix::operator- (
        const Matrix &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        Matrix result(m_width, m_height);
        _MATH(sub, m_size, m_data, other.m_data, result.m_data);

        return result;
}

void Matrix::operator+= (
        const Matrix &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        _MATH(sum, m_size, m_data, other.m_data, m_data);
}

void Matrix::operator-= (
        const Matrix &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        _MATH(sub, m_size, m_data, other.m_data, m_data);
}

Matrix Matrix::operator+ (
        float scalar
) const {
        Matrix result(m_width, m_height);
        _MATH(sum, m_size, m_data, scalar, result.m_data);

        return result;
}

Matrix Matrix::operator- (
        float scalar
) const {
        Matrix result(m_width, m_height);
        _MATH(sub, m_size, m_data, scalar, result.m_data);

        return result;
}

Matrix Matrix::operator* (
        float scalar
) const {
        Matrix result(m_width, m_height);
        _MATH(mul, m_size, m_data, scalar, result.m_data);

        return result;
}

Matrix Matrix::operator/ (
        float scalar
) const {
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Matrix result(m_width, m_height);
        _MATH(div, m_size, m_data, scalar, result.m_data);

        return result;
}

void Matrix::operator+= (
        float scalar
) {
        _MATH(sum, m_size, m_data, scalar, m_data);
}

void Matrix::operator-= (
        float scalar
) {
        _MATH(sub, m_size, m_data, scalar, m_data);
}

void Matrix::operator*= (
        float scalar
) {
        _MATH(mul, m_size, m_data, scalar, m_data);
}

void Matrix::operator/= (
        float scalar
) {
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        _MATH(div, m_size, m_data, scalar, m_data);
}

Vector Matrix::operator* (
        const Vector &vec
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Matrix width and vector size must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_height);
        _MATH(matrixVectorMul, m_width, m_height, m_data, vec.data(), result.data());

        return result;
}
