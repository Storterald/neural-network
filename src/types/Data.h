#pragma once

#include "../math/Base.h"

class Data {
public:
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

        [[nodiscard]] inline float *begin() noexcept
        {
                return m_data;
        }

        [[nodiscard]] inline const float *begin() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] inline float *end() noexcept
        {
                return m_data + m_size;
        }

        [[nodiscard]] inline const float *end() const noexcept
        {
                return m_data + m_size;
        }

        [[nodiscard]] inline uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] inline uint64_t hash() const noexcept
        {
                uint64_t hash{};
                for (uint32_t i { 0 }; i < m_size; i++)
                        hash ^= std::hash<float>{}(m_data[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

                return hash;
        }

        [[nodiscard]] bool operator== (const Data &other) const;

protected:
        uint32_t        m_size  = 0;
        float           *m_data = nullptr;

};

// Hash for Data and Data derived classes, allows for use in hash map as key.
template<>
struct std::hash<Data>
{
        std::size_t inline operator() (const Data &data) const noexcept
        {
                return data.hash();
        }
};