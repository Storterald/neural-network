#pragma once

#include <utility>

#include "Vector.h"

class Matrix : public Data {
public:
        inline Matrix() = default;
        Matrix(uint32_t width, uint32_t height);
        Matrix(std::initializer_list<std::initializer_list<float>> values);

        [[nodiscard]] Ptr<float> operator[] (uint32_t row);
        [[nodiscard]] Ref<float> operator[] (std::pair<uint32_t, uint32_t> position);
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

#ifdef DEBUG_MODE_ENABLED
#include <iostream>
#include <string>
inline std::ostream &operator<< (std::ostream &os, const Matrix &mat)
{
        std::string str;
        for (uint32_t i { 0 }; i < mat.height(); ++i) {
                str += '[';
                for (uint32_t j { 0 }; j < mat.width() - 1; ++j)
                        str += std::to_string(mat.at(i)[j]) + ", ";

                str += std::to_string(mat.at(i)[mat.width() - 1]) + ']';
        }

        os << "[" << str << "]";
        return os;
}
#endif // DEBUG_MODE_ENABLED
