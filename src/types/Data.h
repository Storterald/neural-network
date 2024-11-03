#pragma once

#include <cstdint>

class Data {
public:
        // Cannot be constexpr since this header is compiled in CUDA, and its
        // standard is C++20, while constexpr constructors are in C++23.
        inline Data() = default;
        explicit Data(uint32_t size);

        Data(const Data &other);
        Data &operator= (const Data &other);

        Data(Data &&other) noexcept;
        Data &operator= (Data &&other) noexcept;

        ~Data();

        [[nodiscard]] inline float *data() noexcept
        {
                return m_data;
        }

        [[nodiscard]] inline const float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] inline uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] bool operator== (const Data &other) const;

protected:
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};
