#include <neural-network/types/vector.h>

#include <initializer_list>
#include <algorithm> // std::ranges::any_of
#include <iterator>  // std::data
#include <cstdint>
#include <cstring>   // std::memcpy

#include <neural-network/types/memory.h>
#include <neural-network/types/buf.h>
#include <neural-network/math/math.h>
#include <neural-network/base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

vector::vector(uint32_t size) : buf(size) {}

vector::vector(uint32_t size, const float values[]) : buf(size)
{
        if (!m_device) {
                std::memcpy(this->data().get(), values, m_size * sizeof(float));
                return;
        }

#if BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaMemcpy(this->data().get(), values, m_size * sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

vector::vector(const std::initializer_list<float> &values) :
        buf((uint32_t)values.size()) {

        if (!m_device) {
                std::memcpy(this->data().get(), std::data(values), m_size * sizeof(float));
                return;
        }

#if BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaMemcpy(this->data().get(), std::data(values), m_size * sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

ref<vector::value_type> vector::operator[] (uint32_t i)
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + i);
}

ref<vector::const_value_type> vector::operator[] (uint32_t i) const
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + i);
}

float vector::at(uint32_t i) const
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + i);
}

vector vector::operator+ (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::sum(m_size, *this, other, result);

        return result;
}

vector vector::operator- (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::sub(m_size, *this, other, result);

        return result;
}

vector vector::operator* (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::mul(m_size, *this, other, result);

        return result;
}

vector vector::operator/ (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::div(m_size, *this, other, result);

        return result;
}

void vector::operator+= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::sum(m_size, *this, other, *this);
}

void vector::operator-= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::sub(m_size, *this, other, *this);
}

void vector::operator*= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::mul(m_size, *this, other, *this);
}

void vector::operator/= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(m_size, *this, other, *this);
}

vector vector::operator+ (float scalar) const
{
        vector result(m_size);
        math::sum(m_size, *this, scalar, result);

        return result;
}

vector vector::operator- (float scalar) const
{
        vector result(m_size);
        math::sub(m_size, *this, scalar, result);

        return result;
}

vector vector::operator* (float scalar) const
{
        vector result(m_size);
        math::mul(m_size, *this, scalar, result);

        return result;
}

vector vector::operator/ (float scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::div(m_size, *this, scalar, result);

        return result;
}

void vector::operator+= (float scalar)
{
        math::sum(m_size, *this, scalar, *this);
}

void vector::operator-= (float scalar)
{
        math::sub(m_size, *this, scalar, *this);
}

void vector::operator*= (float scalar)
{
        math::mul(m_size, *this, scalar, *this);
}

void vector::operator/= (float scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(m_size, *this, scalar, *this);
}

vector vector::min(float min) const
{
        vector result(m_size);
        math::min(m_size, *this, min, result);

        return result;
}

vector vector::max(float max) const
{
        vector result(m_size);
        math::max(m_size, *this, max, result);

        return result;
}

vector vector::clamp(float min, float max) const
{
        vector result(m_size);
        math::clamp(m_size, *this, min, max, result);

        return result;
}

vector vector::min(const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::min(m_size, *this, other, result);

        return result;
}

vector vector::max(const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::max(m_size, *this, other, result);

        return result;
}

vector vector::clamp(const vector &min, const vector &max) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != min.m_size || m_size!= max.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(m_size);
        math::clamp(m_size, *this, min, max, result);

        return result;
}

} // namespace nn
