#pragma once

#include <cstdint>
#include <cstddef>  // size_t
#include <cstring>  // std::memcpy

#ifndef _MSC_VER
#include <new>  // std::align_val_t
#endif // !_MSC_VER

#include <neural-network/types/memory.h>
#include <neural-network/base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

enum loc_type {
        keep,
        host,
        device

}; // enum location

template<typename _type>
class buf {
        static_assert(floating_type<_type>, "_type must be a floating_type.");
public:
        using value_type       = _type;
        using const_value_type = std::add_const_t<value_type>;
        using size_type        = uint32_t;
        using pointer          = ptr<value_type>;
        using const_pointer    = ptr<const_value_type>;
        using reference        = ref<value_type>;
        using const_reference  = ref<const_value_type>;
        using difference_type  = typename pointer::difference_type;

        inline buf() = default;
        inline buf(std::nullptr_t) : buf() {}

        inline explicit buf(
                uint32_t size,
                stream_t stream = invalid_stream);

        inline buf(const buf &other);
        inline buf &operator= (const buf &other);

        inline buf(buf &&other) noexcept;
        inline buf &operator= (buf &&other) noexcept;

        inline ~buf();

        [[nodiscard]] inline operator bool () const noexcept;

        [[nodiscard]] inline span<value_type> data(loc_type location = keep, bool update = false);
        [[nodiscard]] inline view<value_type> data(loc_type location = keep) const;

        [[nodiscard]] inline pointer begin();
        [[nodiscard]] inline const_pointer begin() const;
        [[nodiscard]] inline pointer end();
        [[nodiscard]] inline const_pointer end() const;

        [[nodiscard]] inline size_type size() const noexcept;
        [[nodiscard]] inline loc_type location() const noexcept;
        [[nodiscard]] inline stream_t stream() const noexcept;

        inline void move(loc_type location, stream_t stream);

protected:
        value_type *m_data  = nullptr;
        size_type  m_size   = 0;
        bool       m_owning = true;
#ifdef BUILD_CUDA_SUPPORT
        stream_t   m_stream = invalid_stream;
        bool       m_device = false;
#else // BUILD_CUDA_SUPPORT
        static constexpr stream_t m_stream = invalid_stream;
        static constexpr bool     m_device = false;
#endif // !BUILD_CUDA_SUPPORT

        struct not_owned_t {};
        static constexpr not_owned_t not_owned{};
        inline buf(not_owned_t, const buf &self);

private:
        inline value_type *_alloc_cpu(uint32_t count);
        inline void _free_cpu();
        inline void _free() noexcept;

}; // class buf

template<typename _type>
inline buf<_type>::buf(uint32_t size, [[maybe_unused]] stream_t stream) :
        m_size(size)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(stream),
        m_device(stream != invalid_stream)
#endif // BUILD_CUDA_SUPPORT
        {

        if (m_size == 0)
                return;

        if (!m_device) {
                m_data = _alloc_cpu(m_size);
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        m_data = cuda::alloc<value_type>(m_size, m_stream);
        cuda::memset(m_data, 0, m_size * sizeof(value_type), m_stream);
#endif // BUILD_CUDA_SUPPORT
}

template<typename _type>
inline buf<_type>::buf(const buf &other) :
        m_size(other.m_size)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(other.m_stream),
        m_device(other.m_device)
#endif // BUILD_CUDA_SUPPORT
        {

        if (m_size == 0)
                return;

        if (!m_device) {
                m_data = _alloc_cpu(m_size);
                std::memcpy(m_data, other.m_data, m_size * sizeof(value_type));
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        m_data = cuda::alloc<value_type>(m_size, m_stream);
        cuda::memcpy(m_data, other.m_data, m_size * sizeof(value_type),
                cudaMemcpyDeviceToDevice, m_stream);
#endif // BUILD_CUDA_SUPPORT
}

template<typename _type>
inline buf<_type>::buf(buf &&other) noexcept :
        m_data(other.m_data),
        m_size(other.m_size)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(other.m_stream),
        m_device(other.m_device)
#endif // BUILD_CUDA_SUPPORT
        {

        other.m_size = 0;
        other.m_data = nullptr;
}

template<typename _type>
inline buf<_type>::~buf()
{
        _free();
}

template<typename _type>
inline buf<_type> &buf<_type>::operator= (const buf &other)
{
        if (this == &other)
                return *this;

        if (!m_owning)
                return this->operator=((buf &&)other);

#ifdef BUILD_CUDA_SUPPORT
        m_stream = other.m_stream;
#endif // BUILD_CUDA_SUPPORT

        if (m_size != other.m_size || m_device != other.m_device) {
                _free();

                m_size   = other.m_size;
#ifdef BUILD_CUDA_SUPPORT
                m_device = other.m_device;
#endif // BUILD_CUDA_SUPPORT

                if (!m_device)
                        m_data = _alloc_cpu(m_size);
#ifdef BUILD_CUDA_SUPPORT
                else
                        m_data = cuda::alloc<value_type>(m_size, m_stream);
#endif // BUILD_CUDA_SUPPORT
        }

        if (!m_device)
                std::memcpy(m_data, other.m_data, m_size * sizeof(value_type));
#ifdef BUILD_CUDA_SUPPORT
        else
                cuda::memcpy(m_data, other.m_data, m_size * sizeof(value_type),
                        cudaMemcpyDeviceToDevice, m_stream);
#endif // BUILD_CUDA_SUPPORT

        return *this;
}

template<typename _type>
inline buf<_type> &buf<_type>::operator= (buf &&other) noexcept
{
        if (this == &other)
                return *this;

        _free();

        m_size   = other.m_size;
        m_data   = other.m_data;

#ifdef BUILD_CUDA_SUPPORT
        m_stream = other.m_stream;
        m_device = other.m_device;
#endif // BUILD_CUDA_SUPPORT

        other.m_size = 0;
        other.m_data = nullptr;

        return *this;
}

template<typename _type>
inline buf<_type>::operator bool () const noexcept
{
        return m_data != nullptr && m_size != 0;
}

template<typename _type>
inline auto buf<_type>::data(loc_type location, bool update) -> span<value_type>
{
        const bool device = location == keep ? m_device : location == loc_type::device;
        return { m_size, device, m_data, m_device, update, m_stream };
}

template<typename _type>
inline auto buf<_type>::data(loc_type location) const -> view<value_type>
{
        const bool device = location == keep ? m_device : location == loc_type::device;
        return { m_size, device, m_data, m_device, false, m_stream };
}

template<typename _type>
inline auto buf<_type>::begin() -> pointer
{
        return { m_data, m_device };
}

template<typename _type>
inline auto buf<_type>::begin() const -> const_pointer
{
        return { m_data, m_device };
}

template<typename _type>
inline auto buf<_type>::end() -> pointer
{
        return { m_data + m_size, m_device };
}

template<typename _type>
inline auto buf<_type>::end() const -> const_pointer
{
        return { m_data + m_size, m_device };
}

template<typename _type>
inline auto buf<_type>::size() const noexcept -> size_type
{
        return m_size;
}

template<typename _type>
inline loc_type buf<_type>::location() const noexcept
{
        return m_device ? device : host;
}

template<typename _type>
inline stream_t buf<_type>::stream() const noexcept
{
        return m_stream;
}

template<typename _type>
inline void buf<_type>::move([[maybe_unused]] loc_type location, [[maybe_unused]] stream_t stream)
{
#ifdef BUILD_CUDA_SUPPORT
        if (location == (m_device ? device : host))
                return;

        value_type *data;
        if (m_device) {
                data = _alloc_cpu(m_size);
                cuda::memcpy(data, m_data, m_size * sizeof(value_type),
                        cudaMemcpyDeviceToHost, m_stream);
                cuda::free(m_data, m_stream);
                cuda::sync(m_stream);

                m_stream = invalid_stream;
        } else {
                m_stream = stream;

                data = cuda::alloc<value_type>(m_size, m_stream);
                cuda::memcpy(data, m_data, m_size * sizeof(value_type),
                        cudaMemcpyHostToDevice, m_stream);
                _free_cpu();
        }

        m_device = location == device;
        m_stream = stream;
        m_data   = data;
#endif // BUILD_CUDA_SUPPORT
}

template<typename _type>
inline buf<_type>::buf(not_owned_t, const buf &self) :
        m_data(self.m_data),
        m_size(self.m_size),
        m_owning(false)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(self.m_stream),
        m_device(self.m_device)
#endif // BUILD_CUDA_SUPPORT
{}

template<typename _type>
inline auto buf<_type>::_alloc_cpu(uint32_t count) -> value_type *
{
#ifndef _MSC_VER
        return new (std::align_val_t(64)) value_type[count]();
#else // !_MSC_VER
        value_type *out = (value_type *)_aligned_malloc(count * sizeof(value_type), 64);
        memset(out, 0, count * sizeof(value_type));
        return out;
#endif // _MSC_VER
}

template<typename _type>
inline void buf<_type>::_free_cpu()
{
        if (!m_owning)
                return;

#ifndef _MSC_VER
        operator delete[] (m_data, std::align_val_t{64});
#else // !_MSC_VER
        _aligned_free(m_data);
#endif // _MSC_VER
}

template<typename _type>
inline void buf<_type>::_free() noexcept
{
        if (!m_data)
                return;

        if (m_owning) {
                if (!m_device)
                        _free_cpu();
#ifdef BUILD_CUDA_SUPPORT
                else
                        cuda::free(m_data, m_stream);
#endif // BUILD_CUDA_SUPPORT
        }

        m_data = nullptr;
        m_size = 0;
}

} // namespace nn

template<typename T>
struct std::hash<nn::buf<T>> {
        inline size_t operator() (const nn::buf<T> &data) const noexcept
        {
                const float *span = data.view(nn::buf<T>::loc_type::host);

                uint64_t hash = 0;
                for (uint32_t i = 0; i < data.size(); i++)
                        hash ^= std::hash<T>{}(span[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

                return hash;
        }

}; // struct std::hash<buf>
