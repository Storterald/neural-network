#pragma once

#include "Base.h"

class Vector {
        friend class Matrix;

public:
        constexpr Vector() = default;
        explicit Vector(uint32_t size);
        Vector(uint32_t size, const float *data);
        Vector(const Vector &other);
        Vector(Vector &&other) noexcept;
        ~Vector();

        Vector &operator= (const Vector &other) noexcept;
        Vector &operator= (Vector &&other) noexcept;

        [[nodiscard]] Vector operator+ (const Vector &other) const;
        [[nodiscard]] Vector operator* (const Vector &other) const;

        void operator+= (const Vector &other);
        void operator-= (const Vector &other);
        void operator*= (const Vector &other);

        [[nodiscard]] Vector operator* (float scalar) const noexcept;
        void operator/= (float scalar);

        [[nodiscard]] Vector operator* (const float *array) const noexcept;
        [[nodiscard]] Vector operator- (const float *array) const noexcept;

        [[nodiscard]] inline uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] inline float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] inline float &operator[] (uint32_t i) const {
#ifdef DEBUG_MODE_ENABLED
                if (i >= m_size)
                        throw Logger::fatal_error("Vector[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                return m_data[i];
        }

        [[nodiscard]] inline float at(uint32_t i) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (i >= m_size)
                        throw Logger::fatal_error("Vector::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                return m_data[i];
        }

private:
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};