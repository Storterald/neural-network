#pragma once

#include <initializer_list>
#include <cstdint>

#include <neural-network/types/memory.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

class matrix : public buf {
public:
        struct indexer {
                uint32_t row;
                uint32_t column;

        };

        matrix() = default;
        matrix(uint32_t width, uint32_t height, nn::stream stream = invalid_stream);
        matrix(std::initializer_list<std::initializer_list<value_type>> values, nn::stream stream = invalid_stream);

        [[nodiscard]] pointer operator[] (uint32_t row);
        [[nodiscard]] const_pointer operator[] (uint32_t row) const;
        [[nodiscard]] const_pointer at(uint32_t row) const;

#ifndef __CUDACC__  // nvcc does not yet support C++23
        [[nodiscard]] inline reference operator[] (uint32_t row, uint32_t column)
        {
                return this->operator[]({row, column});
        }

        [[nodiscard]] inline const_reference operator[] (uint32_t row, uint32_t column) const
        {
                return this->operator[]({row, column});
        }
#endif // !__CUDACC__

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
        uint32_t m_width  = 0;
        uint32_t m_height = 0;

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
