#include <neural-network/types/buf.h>

#include <cstdint>
#include <cstring>  // std::memcpy

#ifndef _MSC_VER
#include <new>  // std::align_val_t
#endif // !_MSC_VER

#include <neural-network/math/math.h>
#include <neural-network/base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

buf::buf(uint32_t size, [[maybe_unused]] stream_t stream) :
#ifdef BUILD_CUDA_SUPPORT
        m_stream(stream), m_device(stream != invalid_stream),
#endif // BUILD_CUDA_SUPPORT
        m_size(size) {

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

buf::buf(const buf &other) :
#ifdef BUILD_CUDA_SUPPORT
        m_stream(other.m_stream), m_device(other.m_device),
#endif // BUILD_CUDA_SUPPORT
        m_size(other.m_size) {

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

buf::buf(buf &&other) noexcept : m_data(other.m_data),
#ifdef BUILD_CUDA_SUPPORT
        m_stream(other.m_stream), m_device(other.m_device),
#endif // BUILD_CUDA_SUPPORT
        m_size(other.m_size) {

        other.m_size = 0;
        other.m_data = nullptr;
}

buf::~buf()
{
        _free();
}

buf &buf::operator= (const buf &other)
{
        if (this == &other)
                return *this;

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

buf &buf::operator= (buf &&other) noexcept
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

bool buf::operator== (const buf &other) const
{
        if (m_size != other.m_size)
                return false;

        bool ans;
        math::compare(m_size, *this, other, &ans);
        return ans;
}

void buf::move([[maybe_unused]] loc_type location, [[maybe_unused]] stream_t stream)
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

inline buf::value_type *buf::_alloc_cpu(uint32_t count)
{
#ifndef _MSC_VER
        return new (std::align_val_t(64)) value_type[count]();
#else // !_MSC_VER
        value_type *out = (value_type *)_aligned_malloc(count * sizeof(value_type), 64);
        memset(out, 0, count * sizeof(value_type));
        return out;
#endif // _MSC_VER
}

inline void buf::_free_cpu()
{
#ifndef _MSC_VER
        operator delete[] (m_data, std::align_val_t{64});
#else // !_MSC_VER
        _aligned_free(m_data);
#endif // _MSC_VER
}

void buf::_free() noexcept
{
        if (!m_data)
                return;

        if (!m_device)
                _free_cpu();
#ifdef BUILD_CUDA_SUPPORT
        else
                cuda::free(m_data, m_stream);
#endif // BUILD_CUDA_SUPPORT

        m_data = nullptr;
        m_size = 0;
}

} // namespace nn