#pragma once

#include <initializer_list>
#include <cstdint>

#include <neural-network/types/buf.h>
#include <neural-network/base.h>

namespace nn {

class vector : public buf {
public:
        vector() = default;
        explicit vector(uint32_t size, nn::stream stream = invalid_stream);
        vector(uint32_t size, const value_type values[], nn::stream stream = invalid_stream);
        vector(const std::initializer_list<value_type> &values, nn::stream stream = invalid_stream);

        [[nodiscard]] reference operator[] (uint32_t i);
        [[nodiscard]] const_reference operator[] (uint32_t i) const;
        [[nodiscard]] value_type at(uint32_t i) const;

        [[nodiscard]] vector min  (value_type min) const;
        [[nodiscard]] vector max  (value_type max) const;
        [[nodiscard]] vector clamp(value_type min, value_type max) const;

        [[nodiscard]] vector min  (const vector &other) const;
        [[nodiscard]] vector max  (const vector &other) const;
        [[nodiscard]] vector clamp(const vector &min, const vector &max) const;

        [[nodiscard]] vector operator+ (const vector &other) const;
        [[nodiscard]] vector operator- (const vector &other) const;
        [[nodiscard]] vector operator* (const vector &other) const;
        [[nodiscard]] vector operator/ (const vector &other) const;

        void operator+= (const vector &other);
        void operator-= (const vector &other);
        void operator*= (const vector &other);
        void operator/= (const vector &other);

        [[nodiscard]] vector operator+ (value_type scalar) const;
        [[nodiscard]] vector operator- (value_type scalar) const;
        [[nodiscard]] vector operator* (value_type scalar) const;
        [[nodiscard]] vector operator/ (value_type scalar) const;

        void operator+= (value_type scalar);
        void operator-= (value_type scalar);
        void operator*= (value_type scalar);
        void operator/= (value_type scalar);

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
