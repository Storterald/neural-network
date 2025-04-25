#pragma once

#include <cstdint>

#include "../Base.h"
#include "Memory.h"

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

        [[nodiscard]] constexpr bool device() const noexcept
        {
                return m_device;
        }

        [[nodiscard]] inline Span<float> as_span(DataLocation location = KEEP, bool updateOnDestruction = false) const
        {
                return Ptr<float>(m_data, m_device).span(m_size,
                        location == KEEP ? m_device : location != HOST, updateOnDestruction);
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

template<>
struct std::hash<Data> {
        std::size_t operator() (const Data &data) const noexcept
        {
                float *span = data.as_span(Data::HOST);

                uint64_t hash = 0;
                for (uint32_t i = 0; i < data.size(); i++)
                        hash ^= std::hash<float>{}(span[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

                return hash;
        }

}; // struct std::hash<Data>
