#pragma once

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

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
        explicit vector(uint32_t size, loc_type location = KEEP);
        vector(uint32_t size, const value_type values[], loc_type location = KEEP);
        vector(const std::initializer_list<value_type> &values, loc_type location = KEEP);
        vector(uint32_t size, nn::stream stream);

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
