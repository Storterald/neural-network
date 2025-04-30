#pragma once

#include <initializer_list>
#include <cstdint>

#include <neural-network/types/buf.h>
#include <neural-network/base.h>

#ifdef DEBUG_MODE_ENABLED
#include <ostream>
#include <string>
#endif // DEBUG_MODE_ENABLED

namespace nn {

class vector : public buf {
public:
        vector() = default;
        explicit vector(uint32_t size);
        vector(uint32_t size, const float values[]);
        vector(const std::initializer_list<float> &values);

        [[nodiscard]] ref<value_type> operator[] (uint32_t i);
        [[nodiscard]] ref<const_value_type> operator[] (uint32_t i) const;
        [[nodiscard]] float at(uint32_t i) const;

        [[nodiscard]] vector min(float min) const;
        [[nodiscard]] vector max(float max) const;
        [[nodiscard]] vector clamp(float min, float max) const;

        [[nodiscard]] vector min(const vector &other) const;
        [[nodiscard]] vector max(const vector &other) const;
        [[nodiscard]] vector clamp(const vector &min, const vector &max) const;

        [[nodiscard]] vector operator+ (const vector &other) const;
        [[nodiscard]] vector operator- (const vector &other) const;
        [[nodiscard]] vector operator* (const vector &other) const;
        [[nodiscard]] vector operator/ (const vector &other) const;

        void operator+= (const vector &other);
        void operator-= (const vector &other);
        void operator*= (const vector &other);
        void operator/= (const vector &other);

        [[nodiscard]] vector operator+ (float scalar) const;
        [[nodiscard]] vector operator- (float scalar) const;
        [[nodiscard]] vector operator* (float scalar) const;
        [[nodiscard]] vector operator/ (float scalar) const;

        void operator+= (float scalar);
        void operator-= (float scalar);
        void operator*= (float scalar);
        void operator/= (float scalar);

}; // class vector

} // namespace nn

[[nodiscard]] inline nn::vector operator+ (float scalar, const nn::vector &vec) {
        return vec + scalar;
}

[[nodiscard]] inline nn::vector operator- (float scalar, const nn::vector &vec) {
        return vec - scalar;
}

[[nodiscard]] inline nn::vector operator* (float scalar, const nn::vector &vec) {
        return vec * scalar;
}

[[nodiscard]] inline nn::vector operator/ (float scalar, const nn::vector &vec) {
        return vec / scalar;
}

#ifdef DEBUG_MODE_ENABLED
inline std::ostream &operator<< (std::ostream &os, const nn::vector &vec)
{
        std::string str;
        for (uint32_t i = 0; i < vec.size() - 1; ++i)
                str += std::to_string(vec.at(i)) + ", ";

        os << "[" << str << vec.at(vec.size() - 1) << "]";
        return os;
}
#endif // DEBUG_MODE_ENABLED
