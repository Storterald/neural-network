#include <neural-network/types/Vector.h>

#include <initializer_list>
#include <algorithm> // std::ranges::any_of
#include <iterator>  // std::data
#include <cstdint>
#include <cstring>   // std::memcpy

#include <neural-network/types/Memory.h>
#include <neural-network/types/Data.h>
#include <neural-network/math/Math.h>
#include <neural-network/Base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/CudaBase.h>
#endif // BUILD_CUDA_SUPPORT

Vector::Vector(uint32_t size) : Data(size) {}

Vector::Vector(uint32_t size, const float values[]) : Data(size)
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

Vector::Vector(const std::initializer_list<float> &values) :
        Data((uint32_t)values.size()) {

        if (!m_device) {
                std::memcpy(this->data().get(), std::data(values), m_size * sizeof(float));
                return;
        }

#if BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaMemcpy(this->data().get(), std::data(values), m_size * sizeof(float),
                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

Ref<float> Vector::operator[] (uint32_t i)
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + i);
}

float Vector::at(uint32_t i) const
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->data() + i);
}

Vector Vector::operator+ (const Vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::sum(m_size, *this, other, result);

        return result;
}

Vector Vector::operator- (const Vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::sub(m_size, *this, other, result);

        return result;
}

Vector Vector::operator* (const Vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::mul(m_size, *this, other, result);

        return result;
}

Vector Vector::operator/ (const Vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::div(m_size, *this, other, result);

        return result;
}

void Vector::operator+= (const Vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Math::sum(m_size, *this, other, *this);
}

void Vector::operator-= (const Vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Math::sub(m_size, *this, other, *this);
}

void Vector::operator*= (const Vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Math::mul(m_size, *this, other, *this);
}

void Vector::operator/= (const Vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Math::div(m_size, *this, other, *this);
}

Vector Vector::operator+ (float scalar) const
{
        Vector result(m_size);
        Math::sum(m_size, *this, scalar, result);

        return result;
}

Vector Vector::operator- (float scalar) const
{
        Vector result(m_size);
        Math::sub(m_size, *this, scalar, result);

        return result;
}

Vector Vector::operator* (float scalar) const
{
        Vector result(m_size);
        Math::mul(m_size, *this, scalar, result);

        return result;
}

Vector Vector::operator/ (float scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::div(m_size, *this, scalar, result);

        return result;
}

void Vector::operator+= (float scalar)
{
        Math::sum(m_size, *this, scalar, *this);
}

void Vector::operator-= (float scalar)
{
        Math::sub(m_size, *this, scalar, *this);
}

void Vector::operator*= (float scalar)
{
        Math::mul(m_size, *this, scalar, *this);
}

void Vector::operator/= (float scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw LOGGER_EX("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Math::div(m_size, *this, scalar, *this);
}

Vector Vector::min(float min) const
{
        Vector result(m_size);
        Math::min(m_size, *this, min, result);

        return result;
}

Vector Vector::max(float max) const
{
        Vector result(m_size);
        Math::max(m_size, *this, max, result);

        return result;
}

Vector Vector::clamp(float min, float max) const
{
        Vector result(m_size);
        Math::clamp(m_size, *this, min, max, result);

        return result;
}

Vector Vector::min(const Vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::min(m_size, *this, other, result);

        return result;
}

Vector Vector::max(const Vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::max(m_size, *this, other, result);

        return result;
}

Vector Vector::clamp(const Vector &min, const Vector &max) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_size != min.m_size || m_size!= max.m_size)
                throw LOGGER_EX("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        Math::clamp(m_size, *this, min, max, result);

        return result;
}
