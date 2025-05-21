#include <neural-network/types/vector.h>

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

vector::vector(uint32_t size, nn::stream stream) :
        buf(size, stream) {}

vector::vector(uint32_t size, const value_type values[], nn::stream stream) :
        buf(size, stream) {

        if (!m_device) {
                std::memcpy(m_data, values, m_size * sizeof(value_type));
                return;
        }

#if BUILD_CUDA_SUPPORT
        cuda::memcpy(m_data, values, m_size * sizeof(value_type),
                cudaMemcpyHostToDevice, m_stream);
#endif // BUILD_CUDA_SUPPORT
}

vector::vector(const std::initializer_list<value_type> &values, nn::stream stream) :
        buf((uint32_t)values.size(), stream) {

        if (!m_device) {
                std::memcpy(m_data, std::data(values), m_size * sizeof(value_type));
                return;
        }

#if BUILD_CUDA_SUPPORT
        cuda::memcpy(m_data, std::data(values), m_size * sizeof(value_type),
                cudaMemcpyHostToDevice, m_stream);
#endif // BUILD_CUDA_SUPPORT
}

vector::reference vector::operator[] (uint32_t i)
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->begin() + i);
}

vector::const_reference vector::operator[] (uint32_t i) const
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->begin() + i);
}

vector::value_type vector::at(uint32_t i) const
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->begin() + i);
}

vector vector::operator+ (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::sum(m_size, *this, other, result);

        return result;
}

vector vector::operator- (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::sub(m_size, *this, other, result);

        return result;
}

vector vector::operator* (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::mul(m_size, *this, other, result);

        return result;
}

vector vector::operator/ (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::div(m_size, *this, other, result);

        return result;
}

void vector::operator+= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::sum(m_size, *this, other, *this);
}

void vector::operator-= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::sub(m_size, *this, other, *this);
}

void vector::operator*= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::mul(m_size, *this, other, *this);
}

void vector::operator/= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(m_size, *this, other, *this);
}

vector vector::operator+ (value_type scalar) const
{
        vector result(m_size, m_stream);
        math::sum(m_size, *this, scalar, result);

        return result;
}

vector vector::operator- (value_type scalar) const
{
        vector result(m_size, m_stream);
        math::sub(m_size, *this, scalar, result);

        return result;
}

vector vector::operator* (value_type scalar) const
{
        vector result(m_size, m_stream);
        math::mul(m_size, *this, scalar, result);

        return result;
}

vector vector::operator/ (value_type scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::div(m_size, *this, scalar, result);

        return result;
}

void vector::operator+= (value_type scalar)
{
        math::sum(m_size, *this, scalar, *this);
}

void vector::operator-= (value_type scalar)
{
        math::sub(m_size, *this, scalar, *this);
}

void vector::operator*= (value_type scalar)
{
        math::mul(m_size, *this, scalar, *this);
}

void vector::operator/= (value_type scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(m_size, *this, scalar, *this);
}

vector vector::min(value_type min) const
{
        vector result(m_size, m_stream);
        math::min(m_size, *this, min, result);

        return result;
}

vector vector::max(value_type max) const
{
        vector result(m_size, m_stream);
        math::max(m_size, *this, max, result);

        return result;
}

vector vector::clamp(value_type min, float max) const
{
        vector result(m_size, m_stream);
        math::clamp(m_size, *this, min, max, result);

        return result;
}

vector vector::min(const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::min(m_size, *this, other, result);

        return result;
}

vector vector::max(const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::max(m_size, *this, other, result);

        return result;
}

vector vector::clamp(const vector &min, const vector &max) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != min.m_size || m_size!= max.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size, m_stream);
        math::clamp(m_size, *this, min, max, result);

        return result;
}

} // namespace nn
