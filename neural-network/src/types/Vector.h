#pragma once

#include <initializer_list>

#include "Data.h"

class Vector : public Data {
public:
        inline Vector() = default;
        explicit Vector(uint32_t size);
        Vector(uint32_t size, const float values[]);
        Vector(const std::initializer_list<float> &values);

        [[nodiscard]] Ref<float> operator[] (uint32_t i);
        [[nodiscard]] float at(uint32_t i) const;

        [[nodiscard]] Vector min(float min) const;
        [[nodiscard]] Vector max(float max) const;
        [[nodiscard]] Vector clamp(float min, float max) const;

        [[nodiscard]] Vector min(const Vector &other) const;
        [[nodiscard]] Vector max(const Vector &other) const;
        [[nodiscard]] Vector clamp(const Vector &min, const Vector &max) const;

        [[nodiscard]] Vector operator+ (const Vector &other) const;
        [[nodiscard]] Vector operator- (const Vector &other) const;
        [[nodiscard]] Vector operator* (const Vector &other) const;
        [[nodiscard]] Vector operator/ (const Vector &other) const;

        void operator+= (const Vector &other);
        void operator-= (const Vector &other);
        void operator*= (const Vector &other);
        void operator/= (const Vector &other);

        [[nodiscard]] Vector operator+ (float scalar) const;
        [[nodiscard]] Vector operator- (float scalar) const;
        [[nodiscard]] Vector operator* (float scalar) const;
        [[nodiscard]] Vector operator/ (float scalar) const;

        void operator+= (float scalar);
        void operator-= (float scalar);
        void operator*= (float scalar);
        void operator/= (float scalar);

}; // class Vector

[[nodiscard]] inline Vector operator+ (float scalar, const Vector &vec) {
        return vec + scalar;
}

[[nodiscard]] inline Vector operator- (float scalar, const Vector &vec) {
        return vec - scalar;
}

[[nodiscard]] inline Vector operator* (float scalar, const Vector &vec) {
        return vec * scalar;
}

[[nodiscard]] inline Vector operator/ (float scalar, const Vector &vec) {
        return vec / scalar;
}

#ifdef DEBUG_MODE_ENABLED
#include <iostream>
#include <string>
inline std::ostream &operator<< (std::ostream &os, const Vector &vec)
{
        std::string str;
        for (uint32_t i = 0; i < vec.size() - 1; ++i)
                str += std::to_string(vec.at(i)) + ", ";

        os << "[" << str << vec.at(vec.size() - 1) << "]";
        return os;
}
#endif // DEBUG_MODE_ENABLED
