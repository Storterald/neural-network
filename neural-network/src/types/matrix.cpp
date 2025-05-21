#include <neural-network/types/matrix.h>

#include <initializer_list>
#include <iterator>  // std::data
#include <cstdint>
#include <cstring>   // std::memcpy

#include <neural-network/types/memory.h>
#include <neural-network/types/buf.h>
#include <neural-network/math/math.h>
#include <neural-network/base.h>

#ifdef DEBUG_MODE_ENABLED
#include <algorithm> // std::ranges::any_of
#endif // DEBUG_MODE_ENABLED

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

matrix::matrix(uint32_t width, uint32_t height, nn::stream stream) :
        buf(width * height, stream),
        m_width(width),
        m_height(height) {}

matrix::matrix(std::initializer_list<std::initializer_list<value_type>> values, nn::stream stream) :
        buf((uint32_t)(values.begin()->size() * values.size()), stream),
        m_width((uint32_t)values.begin()->size()),
        m_height((uint32_t)values.size()) {

#ifdef DEBUG_MODE_ENABLED
        if (std::ranges::any_of(values, [this](auto &vs) -> bool { return vs.size() != m_width; }))
                throw fatal_error("All initializer lists passed to Matrix() must have the same size.");
#endif // DEBUG_MODE_ENABLED

        if (!m_device) {
                for (uint32_t i = 0; i < m_height; ++i)
                        std::memcpy(
                                this->operator[](i).get(),
                                std::data(std::data(values)[i]),
                                m_width * sizeof(value_type));
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        const size_t copySize = m_width * sizeof(value_type);
        for (uint32_t i = 0; i < m_height; ++i) {
                value_type *dst = (*this)[i].get();
                const_value_type *src = std::data(std::data(values)[i]);
                cuda::memcpy(dst, src, copySize, cudaMemcpyHostToDevice, m_stream);
        }

#endif // BUILD_CUDA_SUPPORT
}

matrix::pointer matrix::operator[](uint32_t row)
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw fatal_error("Matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width;
        return this->begin() + offset;
}

matrix::const_pointer matrix::operator[](uint32_t row) const
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw fatal_error("Matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width;
        return this->begin() + offset;
}

matrix::const_pointer matrix::at(uint32_t row) const
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height)
                throw fatal_error("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width;
        return this->begin() + offset;
}

matrix::reference matrix::operator[] (indexer position)
{
#ifdef DEBUG_MODE_ENABLED
        if (position.row >= m_height || position.column >= m_width)
                throw fatal_error("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)position.row * m_width + position.column;
        return *(this->begin() + offset);
}

matrix::const_reference matrix::operator[] (indexer position) const
{
#ifdef DEBUG_MODE_ENABLED
        if (position.row >= m_height || position.column >= m_width)
                throw fatal_error("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)position.row * m_width + position.column;
        return *(this->begin() + offset);
}

matrix::value_type matrix::at(uint32_t row, uint32_t column) const
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height || column >= m_width)
                throw fatal_error("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = (difference_type)row * m_width + column;
        return *(this->begin() + offset);
}

matrix matrix::operator+ (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height, m_stream);
        math::sum(m_size, *this, other, result);

        return result;
}

matrix matrix::operator- (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height, m_stream);
        math::sub(m_size, *this, other, result);

        return result;
}

void matrix::operator+= (const matrix &other)
        {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        math::sum(m_size, *this, other, *this);
}

void matrix::operator-= (const matrix &other)
        {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        math::sub(m_size, *this, other, *this);
}

matrix matrix::operator+ (value_type scalar) const
{
        matrix result(m_width, m_height, m_stream);
        math::sum(m_size, *this, scalar, result);

        return result;
}

matrix matrix::operator- (value_type scalar) const
{
        matrix result(m_width, m_height, m_stream);
        math::sub(m_size, *this, scalar, result);

        return result;
}

matrix matrix::operator* (value_type scalar) const
{
        matrix result(m_width, m_height, m_stream);
        math::mul(m_size, *this, scalar, result);

        return result;
}

matrix matrix::operator/ (value_type scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height, m_stream);
        math::div(m_size, *this, scalar, result);

        return result;
}

void matrix::operator+= (value_type scalar)
{
        math::sum(m_size, *this, scalar, *this);
}

void matrix::operator-= (value_type scalar)
{
        math::sub(m_size, *this, scalar, *this);
}

void matrix::operator*= (value_type scalar)
{
        math::mul(m_size, *this, scalar, *this);
}

void matrix::operator/= (value_type scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(m_size, *this, scalar, *this);
}

vector matrix::operator* (const vector &vec) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw fatal_error("Matrix width and vector size must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_height, m_stream);
        math::matvec_mul(m_width, m_height, *this, vec, result);

        return result;
}

} // namespace nn
