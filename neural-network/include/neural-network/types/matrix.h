#pragma once

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

#include <initializer_list>
#include <cstdint>

#include <neural-network/types/memory.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

#ifdef DEBUG_MODE_ENABLED
#include <ostream>
#include <string>
#endif // DEBUG_MODE_ENABLED

namespace nn {

class matrix : public buf {
public:
        struct indexer {
                uint32_t row;
                uint32_t column;

        };

        matrix() = default;
        matrix(uint32_t width, uint32_t height, loc_type location = KEEP);
        matrix(std::initializer_list<std::initializer_list<value_type>> values, loc_type location = KEEP);
        matrix(uint32_t width, uint32_t height, nn::stream stream);

        [[nodiscard]] pointer operator[] (uint32_t row);
        [[nodiscard]] const_pointer operator[] (uint32_t row) const;
        [[nodiscard]] const_pointer at(uint32_t row) const;

        [[nodiscard]] reference operator[] (indexer position);
        [[nodiscard]] const_reference operator[] (indexer position) const;
        [[nodiscard]] value_type at(uint32_t row, uint32_t column) const;

        [[nodiscard]] inline size_type width() const noexcept
        {
                return m_width;
        }

        [[nodiscard]] inline size_type height() const noexcept
        {
                return m_height;
        }

        [[nodiscard]] matrix operator+ (const matrix &other) const;
        [[nodiscard]] matrix operator- (const matrix &other) const;

        void operator+= (const matrix &other);
        void operator-= (const matrix &other);

        [[nodiscard]] matrix operator+ (value_type scalar) const;
        [[nodiscard]] matrix operator- (value_type scalar) const;
        [[nodiscard]] matrix operator* (value_type scalar) const;
        [[nodiscard]] matrix operator/ (value_type scalar) const;

        void operator+= (value_type scalar);
        void operator-= (value_type scalar);
        void operator*= (value_type scalar);
        void operator/= (value_type scalar);

        [[nodiscard]] vector operator* (const vector &vec) const;

private:
        uint32_t        m_width  = 0;
        uint32_t        m_height = 0;

}; // class matrix

} // namespace nn

[[nodiscard]] inline nn::matrix operator+ (float scalar, const nn::matrix &mat) {
        return mat + scalar;
}

[[nodiscard]] inline nn::matrix operator- (float scalar, const nn::matrix &mat) {
        return mat - scalar;
}

[[nodiscard]] inline nn::matrix operator* (float scalar, const nn::matrix &mat) {
        return mat * scalar;
}

[[nodiscard]] inline nn::matrix operator/ (float scalar, const nn::matrix &mat) {
        return mat / scalar;
}

#ifdef DEBUG_MODE_ENABLED
inline std::ostream &operator<< (std::ostream &os, const nn::matrix &mat)
{
        std::string str;
        for (uint32_t i = 0; i < mat.height(); ++i) {
                str += '[';
                for (uint32_t j = 0; j < mat.width() - 1; ++j)
                        str += std::to_string(mat.at(i, j)) + ", ";

                str += std::to_string(mat.at(i, mat.width() - 1)) + ']';
        }

        os << "[" << str << "]";
        return os;
}
#endif // DEBUG_MODE_ENABLED
