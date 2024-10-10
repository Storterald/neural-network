#pragma once

#include "Vector.h"

class Matrix {
        friend class Vector;

public:
        constexpr Matrix() = default;
        Matrix(uint32_t width, uint32_t height, float value = 0.0f);
        Matrix(const Matrix &other);
        Matrix(Matrix &&other) noexcept;
        ~Matrix();

        Matrix &operator= (const Matrix &other) noexcept;
        Matrix &operator= (Matrix &&other) noexcept;

        void operator+= (const Matrix &other);
        void operator-= (const Matrix &other);

        // Matrix product visualizer:
        // http://matrixmultiplication.xyz
        [[nodiscard]] Vector operator* (const Vector &vec) const;
        [[nodiscard]] Matrix operator* (float scalar) const noexcept;

        [[nodiscard]] inline uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] inline uint32_t width() const noexcept
        {
                return m_width;
        }

        [[nodiscard]] inline uint32_t height() const noexcept
        {
                return m_height;
        }

        [[nodiscard]] inline float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] constexpr float *operator[] (uint32_t row)
        {
#ifdef DEBUG_MODE_ENABLED
                if (row >= m_height)
                        throw Logger::fatal_error("Matrix[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                return m_data + row * m_width;
        }

        [[nodiscard]] constexpr const float *at(uint32_t row) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (row >= m_height)
                        throw Logger::fatal_error("Matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                return m_data + row * m_width;
        }

private:
        uint32_t m_width { 0 };
        uint32_t m_height { 0 };
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};

#ifdef DEBUG_MODE_ENABLED
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