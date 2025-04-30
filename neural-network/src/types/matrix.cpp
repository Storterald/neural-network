#include <neural-network/types/matrix.h>

#include <initializer_list>
#include <algorithm> // std::ranges::any_of
#include <iterator>  // std::data
#include <cstdint>
#include <cstring>   // std::memcpy

#include <neural-network/types/memory.h>
#include <neural-network/utils/logger.h>
#include <neural-network/types/buf.h>
#include <neural-network/math/math.h>
#include <neural-network/base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

matrix::matrix(uint32_t width, uint32_t height) :
        buf(width * height),
        m_width(width),
        m_height(height) {}

matrix::matrix(std::initializer_list<std::initializer_list<float>> values) :
        buf((uint32_t)(values.begin()->size() * values.size())),
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

ptr<matrix::value_type> matrix::operator[](uint32_t row)
{
        using difference_type = ptr<const_value_type>::difference_type;

#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw LOGGER_EX("Matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width;
        return this->data() + offset;
}

ptr<matrix::const_value_type> matrix::operator[](uint32_t row) const
{
        using difference_type = ptr<const_value_type>::difference_type;

#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw LOGGER_EX("Matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width;
        return this->data() + offset;
}

ptr<matrix::const_value_type> matrix::at(uint32_t row) const
{
        using difference_type = ptr<const_value_type>::difference_type;

#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width;
        return this->data() + offset;
}

ref<matrix::value_type> matrix::operator[] (indexer position)
{
        using difference_type = ptr<const_value_type>::difference_type;

#ifdef DEBUG_MODE_ENABLED
        if (position.row >= m_height || position.column >= m_width)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)position.row * m_width + position.column;
        return *(this->data() + offset);
}

ref<matrix::const_value_type> matrix::operator[] (indexer position) const
{
        using difference_type = ptr<const_value_type>::difference_type;

#ifdef DEBUG_MODE_ENABLED
        if (position.row >= m_height || position.column >= m_width)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)position.row * m_width + position.column;
        return *(this->data() + offset);
}

float matrix::at(uint32_t row, uint32_t height) const
{
        using difference_type = ptr<const_value_type>::difference_type;

#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height || height >= m_width)
                throw LOGGER_EX("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width + height;
        return *(this->data() + offset);
}

matrix matrix::operator+ (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height);
        math::sum(m_size, *this, other, result);

        return result;
}

matrix matrix::operator- (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height);
        math::sub(m_size, *this, other, result);

        return result;
}

void matrix::operator+= (const matrix &other)
        {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        math::sum(m_size, *this, other, *this);
}

void matrix::operator-= (const matrix &other)
        {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw LOGGER_EX("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        math::sub(m_size, *this, other, *this);
}

matrix matrix::operator+ (float scalar) const
{
        matrix result(m_width, m_height);
        math::sum(m_size, *this, scalar, result);

        return result;
}

matrix matrix::operator- (float scalar) const
{
        matrix result(m_width, m_height);
        math::sub(m_size, *this, scalar, result);

        return result;
}

matrix matrix::operator* (float scalar) const
{
        matrix result(m_width, m_height);
        math::mul(m_size, *this, scalar, result);

        return result;
}

matrix matrix::operator/ (float scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height);
        math::div(m_size, *this, scalar, result);

        return result;
}

void matrix::operator+= (float scalar)
{
        math::sum(m_size, *this, scalar, *this);
}

void matrix::operator-= (float scalar)
{
        math::sub(m_size, *this, scalar, *this);
}

void matrix::operator*= (float scalar)
{
        math::mul(m_size, *this, scalar, *this);
}

void matrix::operator/= (float scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(m_size, *this, scalar, *this);
}

vector matrix::operator* (const vector &vec) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw LOGGER_EX("Matrix width and vector size must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_height);
        math::matvec_mul(m_width, m_height, *this, vec, result);

        return result;
}

} // namespace nn
