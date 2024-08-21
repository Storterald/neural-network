#pragma once

#include <cstdint>

#include "../Base.h"

#ifdef DEBUG_MODE_ENABLED
#include "../utils/Logger.h"
#endif // DEBUG_MODE_ENABLED

class Vector {
        friend class Matrix;

public:
        constexpr Vector() = default;
        Vector(uint32_t size);
        Vector(const Vector &other);
        Vector(Vector &&other) noexcept;

        ~Vector();

        Vector &operator= (const Vector &other) noexcept;
        Vector &operator= (Vector &&other) noexcept;
        Vector operator+ (const Vector &other) const;

        void operator+= (const Vector &other) const;
        void operator*= (const Vector &other) const;
        void operator/= (float scalar) const;

        Vector operator* (float scalar) const noexcept;
        Vector operator* (const float *array) const noexcept;
        Vector operator- (const float *array) const noexcept;

        [[nodiscard]] constexpr float operator[] (uint32_t i) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (i >= m_size)
                        throw Logger::fatal_error("[] Vector access index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                return m_data[i];
        }

        [[nodiscard]] constexpr float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] constexpr uint32_t size() const noexcept
        {
                return m_size;
        }

private:
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};