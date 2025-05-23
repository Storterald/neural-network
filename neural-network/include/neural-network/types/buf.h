#pragma once

#include <functional> // std::hash<float>
#include <cstdint>
#include <cstddef>    // size_t

#include <neural-network/types/memory.h>
#include <neural-network/base.h>

namespace nn {

class buf {
public:
        using value_type = float;
        using const_value_type = const float;
        using size_type = uint32_t;
        using pointer = ptr<value_type>;
        using const_pointer = ptr<const_value_type>;
        using reference = ref<value_type>;
        using const_reference = ref<const_value_type>;
        using difference_type = pointer::difference_type;

        enum loc_type {
                KEEP,
                HOST,
                DEVICE

        }; // enum location

        buf() = default;
        /**
         * A invalid stream (null) will allocate the buffer on the cpu.
         */
        explicit buf(uint32_t size, nn::stream stream = invalid_stream);

        buf(const buf &other);
        buf &operator= (const buf &other);

        buf(buf &&other) noexcept;
        buf &operator= (buf &&other) noexcept;

        ~buf();

        [[nodiscard]] bool operator== (const buf &other) const;

        [[nodiscard]] inline span<value_type> data(loc_type location = KEEP, bool update = false)
        {
                const bool device = location == KEEP ? m_device : location == DEVICE;
                return { m_size, device, m_data, m_device, update, m_stream };
        }

        [[nodiscard]] inline view<value_type> view(loc_type location = KEEP) const
        {
                const bool device = location == KEEP ? m_device : location == DEVICE;
                return { m_size, device, m_data, m_device, false, m_stream };
        }

        [[nodiscard]] inline pointer begin()
        {
                return { m_data, m_device };
        }

        [[nodiscard]] inline const_pointer begin() const
        {
                return { m_data, m_device };
        }

        [[nodiscard]] inline pointer end()
        {
                return { m_data + m_size, m_device };
        }

        [[nodiscard]] inline const_pointer end() const
        {
                return { m_data + m_size, m_device };
        }

        [[nodiscard]] constexpr size_type size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] constexpr loc_type location() const noexcept
        {
                return m_device ? DEVICE : HOST;
        }

        [[nodiscard]] constexpr stream stream() const noexcept
        {
                return m_stream;
        }

        void move(loc_type location, ::nn::stream stream);

protected:
        float                              *m_data = nullptr;

#ifdef BUILD_CUDA_SUPPORT
        nn::stream                         m_stream = invalid_stream;
        bool                               m_device = false;
#else // BUILD_CUDA_SUPPORT
        static constexpr nn::stream        m_stream = invalid_stream;
        static constexpr bool              m_device = false;
#endif // BUILD_CUDA_SUPPORT

        size_type                          m_size = 0;

private:
        void _free() noexcept;

}; // class buf

} // namespace nn

template<>
struct std::hash<nn::buf> {
        size_t operator() (const nn::buf &data) const noexcept
        {
                const float *span = data.view(nn::buf::HOST);

                uint64_t hash = 0;
                for (uint32_t i = 0; i < data.size(); i++)
                        hash ^= std::hash<float>{}(span[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

                return hash;
        }

}; // struct std::hash<buf>
