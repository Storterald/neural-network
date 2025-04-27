#pragma once

#include <initializer_list>
#include <cstdint>

#include <neural-network/types/Memory.h>
#include <neural-network/types/Vector.h>
#include <neural-network/Base.h>

#ifdef DEBUG_MODE_ENABLED
#include <iostream>
#include <string>
#endif // DEBUG_MODE_ENABLED

class Matrix : public Data {
public:
        struct Indexer {
                uint32_t row;
                uint32_t column;

        };

        inline Matrix() = default;
        Matrix(uint32_t width, uint32_t height);
        Matrix(std::initializer_list<std::initializer_list<float>> values);

        [[nodiscard]] Ptr<float> operator[] (uint32_t row);
        [[nodiscard]] Ref<float> operator[] (Indexer position);
        [[nodiscard]] Ptr<float> at(uint32_t row) const;
        [[nodiscard]] float at(uint32_t row, uint32_t height) const;

        [[nodiscard]] inline uint32_t width() const noexcept
        {
                return m_width;
        }

        [[nodiscard]] inline uint32_t height() const noexcept
        {
                return m_height;
        }

        [[nodiscard]] Matrix operator+ (const Matrix &other) const;
        [[nodiscard]] Matrix operator- (const Matrix &other) const;

        void operator+= (const Matrix &other);
        void operator-= (const Matrix &other);

        [[nodiscard]] Matrix operator+ (float scalar) const;
        [[nodiscard]] Matrix operator- (float scalar) const;
        [[nodiscard]] Matrix operator* (float scalar) const;
        [[nodiscard]] Matrix operator/ (float scalar) const;

        void operator+= (float scalar);
        void operator-= (float scalar);
        void operator*= (float scalar);
        void operator/= (float scalar);

        [[nodiscard]] Vector operator* (const Vector &vec) const;

private:
        uint32_t        m_width  = 0;
        uint32_t        m_height = 0;

}; // class Matrix

[[nodiscard]] inline Matrix operator+ (float scalar, const Matrix &mat) {
        return mat + scalar;
}

[[nodiscard]] inline Matrix operator- (float scalar, const Matrix &mat) {
        return mat - scalar;
}

[[nodiscard]] inline Matrix operator* (float scalar, const Matrix &mat) {
        return mat * scalar;
}

[[nodiscard]] inline Matrix operator/ (float scalar, const Matrix &mat) {
        return mat / scalar;
}

#ifdef DEBUG_MODE_ENABLED
inline std::ostream &operator<< (std::ostream &os, const Matrix &mat)
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
