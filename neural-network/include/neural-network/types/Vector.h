#pragma once

#include <initializer_list>
#include <cstdint>

#include <neural-network/types/Data.h>
#include <neural-network/Base.h>

#ifdef DEBUG_MODE_ENABLED
#include <iostream>
#include <string>
#endif // DEBUG_MODE_ENABLED

NN_BEGIN

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

NN_END

[[nodiscard]] inline NN Vector operator+ (float scalar, const NN Vector &vec) {
        return vec + scalar;
}

[[nodiscard]] inline NN Vector operator- (float scalar, const NN Vector &vec) {
        return vec - scalar;
}

[[nodiscard]] inline NN Vector operator* (float scalar, const NN Vector &vec) {
        return vec * scalar;
}

[[nodiscard]] inline NN Vector operator/ (float scalar, const NN Vector &vec) {
        return vec / scalar;
}

#ifdef DEBUG_MODE_ENABLED
inline std::ostream &operator<< (std::ostream &os, const NN Vector &vec)
{
        std::string str;
        for (uint32_t i = 0; i < vec.size() - 1; ++i)
                str += std::to_string(vec.at(i)) + ", ";

        os << "[" << str << vec.at(vec.size() - 1) << "]";
        return os;
}
#endif // DEBUG_MODE_ENABLED
