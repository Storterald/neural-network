#pragma once

#include <functional> // std::hash<float>
#include <cstdint>
#include <cstddef>    // size_t

#include <neural-network/types/Memory.h>

class Data {
public:
        enum DataLocation {
                KEEP,
                HOST,
                DEVICE
        };

        inline Data() = default;
        explicit Data(uint32_t size);

        Data(const Data &other);
        Data &operator= (const Data &other);

        Data(Data &&other) noexcept;
        Data &operator= (Data &&other) noexcept;

        ~Data();

        [[nodiscard]] bool operator== (const Data &other) const;

        [[nodiscard]] inline Ptr<float> data() const
        {
                return { m_data, m_device };
        }

        [[nodiscard]] inline Ptr<float> begin() const
        {
                return { m_data, m_device };
        }

        [[nodiscard]] inline Ptr<float> end() const
        {
                return { m_data + m_size, m_device };
        }

        [[nodiscard]] constexpr uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] inline Span<float> as_span(DataLocation location = KEEP, bool updateOnDestruction = false) const
        {
                const bool device = location == KEEP ? m_device : location == DEVICE;
                return { m_size, device, m_data, m_device, updateOnDestruction };
        }

        [[nodiscard]] constexpr DataLocation location() const noexcept
        {
                return m_device ? DEVICE : HOST;
        }

protected:
        uint32_t              m_size   = 0;

#ifdef BUILD_CUDA_SUPPORT
        bool                  m_device = false;
#else // BUILD_CUDA_SUPPORT
        static constexpr bool m_device = false;
#endif // BUILD_CUDA_SUPPORT

private:
        float                 *m_data  = nullptr;

}; // class Data

namespace std {

template<>
struct hash<Data> {
        size_t operator() (const Data &data) const noexcept
        {
                const float *span = data.as_span(Data::HOST);

                uint64_t hash = 0;
                for (uint32_t i = 0; i < data.size(); i++)
                        hash ^= std::hash<float>{}(span[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

                return hash;
        }

}; // struct hash<Data>

} // namespace std
