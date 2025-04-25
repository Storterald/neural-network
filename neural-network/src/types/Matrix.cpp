#include "Matrix.h"

#include "../math/Math.h"

Matrix::Matrix(uint32_t width, uint32_t height) :
        Data(width * height),
        m_width(width),
        m_height(height) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> values) :
        Data((uint32_t)(values.begin()->size() * values.size())),
        m_width((uint32_t)values.begin()->size()),
        m_height((uint32_t)values.size()) {

#ifdef DEBUG_MODE_ENABLED
        if (std::ranges::any_of(values, [this](auto &vs) -> bool { return vs.size() != m_width; }))
                throw LOGGER_EX("All initializer lists passed to Matrix() must have the same size.");
#endif // DEBUG_MODE_ENABLED

        if (!m_device) {
                for (uint32_t i = 0; i < m_height; ++i)
                        std::memcpy(
                                this->operator[](i).get(),
                                std::data(std::data(values)[i]),
                                m_width * sizeof(float));
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        for (uint32_t i = 0; i < m_height; ++i)
                CUDA_CHECK_ERROR(cudaMemcpy(
                        this->operator[](i).get(),
                        std::data(std::data(values)[i]),
                        m_size * sizeof(float), cudaMemcpyHostToDevice),
                        "Failed to copy data to the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

Ptr<float> Matrix::operator[](uint32_t row)
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw LOGGER_EX("Matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return this->data() + row * m_width;
}

Ptr<float> Matrix::at(uint32_t row) const
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return this->data() + row * m_width;
}

Ref<float> Matrix::operator[] (std::pair<uint32_t, uint32_t> position)
{
#ifdef DEBUG_MODE_ENABLED
        if (position.first >= m_height || position.second >= m_width)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + position.first * m_width + position.second);
}

float Matrix::at(uint32_t row, uint32_t height) const
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height || height >= m_width)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + row * m_width + height);
}

Matrix Matrix::operator+ (const Matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        Matrix result(m_width, m_height);
        Math::sum(m_size, *this, other, result);

        return result;
}

Matrix Matrix::operator- (const Matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        Matrix result(m_width, m_height);
        Math::sub(m_size, *this, other, result);

        return result;
}

void Matrix::operator+= (const Matrix &other)
        {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        Math::sum(m_size, *this, other, *this);
}

void Matrix::operator-= (const Matrix &other)
        {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        Math::sub(m_size, *this, other, *this);
}

Matrix Matrix::operator+ (float scalar) const
{
        Matrix result(m_width, m_height);
        Math::sum(m_size, *this, scalar, result);

        return result;
}

Matrix Matrix::operator- (float scalar) const
{
        Matrix result(m_width, m_height);
        Math::sub(m_size, *this, scalar, result);

        return result;
}

Matrix Matrix::operator* (float scalar) const
{
        Matrix result(m_width, m_height);
        Math::mul(m_size, *this, scalar, result);

        return result;
}

Matrix Matrix::operator/ (float scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Matrix result(m_width, m_height);
        Math::div(m_size, *this, scalar, result);

        return result;
}

void Matrix::operator+= (float scalar)
{
        Math::sum(m_size, *this, scalar, *this);
}

void Matrix::operator-= (float scalar)
{
        Math::sub(m_size, *this, scalar, *this);
}

void Matrix::operator*= (float scalar)
{
        Math::mul(m_size, *this, scalar, *this);
}

void Matrix::operator/= (float scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Math::div(m_size, *this, scalar, *this);
}

Vector Matrix::operator* (const Vector &vec) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw LOGGER_EX("Matrix width and vector size must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_height);
        Math::matvec_mul(m_width, m_height, *this, vec, result);

        return result;
}
