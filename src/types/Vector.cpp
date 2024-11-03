#include "Vector.h"

#include <algorithm>

#include "Base.h"

Vector::Vector(
        uint32_t size
) : Data(size) {}

Vector::Vector(
        uint32_t size,
        const float values[]
) : Data(size) {
        std::memcpy(m_data, values, m_size * sizeof(float));
}

Vector::Vector(
        const std::initializer_list<float> &values
) : Data(values.size()) {
        std::memcpy(m_data, values.begin(), m_size * sizeof(float));
}


float &Vector::operator[] (uint32_t i)
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return m_data[i];
}

float Vector::at(uint32_t i) const
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return m_data[i];
}

Vector Vector::operator+ (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(sum, m_size, m_data, other.m_data, result.m_data);

        return result;
}

Vector Vector::operator- (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(sub, m_size, m_data, other.m_data, result.m_data);

        return result;
}

Vector Vector::operator* (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(mul, m_size, m_data, other.m_data, result.m_data);

        return result;
}

Vector Vector::operator/ (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
        if (std::any_of(other.data(), other.data() + other.size(), [](float v) -> bool { return v == 0.0f; }))
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(div, m_size, m_data, other.m_data, result.m_data);

        return result;
}

void Vector::operator+= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        _MATH(sum, m_size, m_data, other.m_data, m_data);
}

void Vector::operator-= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        _MATH(sub, m_size, m_data, other.m_data, m_data);
}

void Vector::operator*= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        _MATH(mul, m_size, m_data, other.m_data, m_data);
}

void Vector::operator/= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
        if (std::any_of(other.data(), other.data() + other.size(), [](float v) -> bool { return v == 0.0f; }))
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        _MATH(div, m_size, m_data, other.m_data, m_data);
}

Vector Vector::operator+ (
        float scalar
) const {
        Vector result(m_size);
        _MATH(sum, m_size, m_data, scalar, result.m_data);

        return result;
}

Vector Vector::operator- (
        float scalar
) const {
        Vector result(m_size);
        _MATH(sub, m_size, m_data, scalar, result.m_data);

        return result;
}

Vector Vector::operator* (
        float scalar
) const {
        Vector result(m_size);
        _MATH(mul, m_size, m_data, scalar, result.m_data);

        return result;
}

Vector Vector::operator/ (
        float scalar
) const {
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(div, m_size, m_data, scalar, result.m_data);

        return result;
}

void Vector::operator+= (
        float scalar
) {
        _MATH(sum, m_size, m_data, scalar, m_data);
}

void Vector::operator-= (
        float scalar
) {
        _MATH(sub, m_size, m_data, scalar, m_data);
}

void Vector::operator*= (
        float scalar
) {
        _MATH(mul, m_size, m_data, scalar, m_data);
}

void Vector::operator/= (
        float scalar
) {
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        _MATH(div, m_size, m_data, scalar, m_data);
}

Vector Vector::min(
        float min
) const {
        Vector result(m_size);
        _MATH(min, m_size, m_data, min, result.m_data);

        return result;
}

Vector Vector::max(
        float max
) const {
        Vector result(m_size);
        _MATH(max, m_size, m_data, max, result.m_data);

        return result;
}

Vector Vector::clamp(
        float min,
        float max
) const {
        Vector result(m_size);
        _MATH(clamp, m_size, m_data, min, max, result.m_data);

        return result;
}

Vector Vector::min(
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(min, m_size, m_data, other.m_data, result.m_data);

        return result;
}

Vector Vector::max(
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(max, m_size, m_data, other.m_data, result.m_data);

        return result;
}

Vector Vector::clamp(
        const Vector &min,
        const Vector &max
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != min.m_size || m_size!= max.m_size)
                throw Logger::fatal_error(LOGGER_PREF(FATAL) + "Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        Vector result(m_size);
        _MATH(clamp, m_size, m_data, min.m_data, max.m_data, result.m_data);

        return result;
}
