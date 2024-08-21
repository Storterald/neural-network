#pragma once

#include "Vector.h"

class Matrix {
        friend class Vector;

public:
        constexpr Matrix() = default;
        Matrix(uint32_t width, uint32_t height);
        Matrix(const Matrix &other);
        Matrix(Matrix &&other) noexcept;

        ~Matrix();

        Matrix &operator= (const Matrix &other) noexcept;
        Matrix &operator= (Matrix &&other) noexcept;

        void operator+= (const Matrix &other);

        // Matrix product visualizer:
        // http://matrixmultiplication.xyz
        Vector operator* (const Vector &vec) const;
        Matrix operator* (float scalar) const noexcept;

        // Returns a pointer to the start of the given row
        constexpr float *operator[] (uint32_t row) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (row >= m_height)
                        throw Logger::fatal_error("[] Matrix access index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                return m_data + row * m_width;
        }

        [[nodiscard]] constexpr float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] constexpr uint32_t width() const noexcept
        {
                return m_width;
        }

        [[nodiscard]] constexpr uint32_t height() const noexcept
        {
                return m_height;
        }

        [[nodiscard]] constexpr uint32_t size() const noexcept
        {
                return m_size;
        }

private:
        uint32_t m_width { 0 };
        uint32_t m_height { 0 };
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};