#include <neural-network/types/buf.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

#include <cstdint>
#include <cstring> // std::memcpy

#include <neural-network/math/math.h>
#include <neural-network/base.h>

namespace nn {

buf::buf(uint32_t size, [[maybe_unused]] loc_type location) : m_size(size)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(0), m_device(location == KEEP ? size >= CUDA_MINIMUM : location == DEVICE)
#endif // BUILD_CUDA_SUPPORT
{
        if (!m_device) {
                m_data = new float[m_size]();
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaMallocAsync(&m_data, m_size * sizeof(float), m_stream),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_data, 0, m_size * sizeof(float), m_stream),
                "Failed to set memory in the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

buf::buf(uint32_t size, [[maybe_unused]] nn::stream stream) : m_size(size)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(stream), m_device(stream != 0 || size >= CUDA_MINIMUM)
#endif // BUILD_CUDA_SUPPORT
{

        if (!m_device) {
                m_data = new float[m_size]();
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaMallocAsync(&m_data, m_size * sizeof(float), m_stream),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_data, 0, m_size * sizeof(float), m_stream),
                "Failed to set memory in the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

buf::buf(const buf &other) : m_size(other.m_size)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(other.m_stream), m_device(other.m_device)
#endif // BUILD_CUDA_SUPPORT
{
        if (!m_device) {
                m_data = new float[m_size];
                std::memcpy(m_data, other.m_data, m_size * sizeof(float));
                return;
        }

#ifdef BUILD_CUDA_SUPPORT
        CUDA_CHECK_ERROR(cudaMallocAsync(&m_data, m_size * sizeof(float), m_stream),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float),
                cudaMemcpyDeviceToDevice, m_stream), "Failed to copy data in the GPU.");
#endif // BUILD_CUDA_SUPPORT
}

buf::buf(buf &&other) noexcept : m_size(other.m_size), m_data(other.m_data)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(other.m_stream), m_device(other.m_device)
#endif // BUILD_CUDA_SUPPORT
{
        other.m_size = 0;
        other.m_data = nullptr;
}

buf::~buf()
{
        if (!m_data)
                return;

        if (!m_device)
                delete [] m_data;
#ifdef BUILD_CUDA_SUPPORT
        else
                CUDA_CHECK_ERROR(cudaFreeAsync(m_data, m_stream), "Failed to free GPU memory.");
#endif // BUILD_CUDA_SUPPORT

        m_data = nullptr;
        m_size = 0;
}

buf &buf::operator= (const buf &other)
{
        if (this == &other)
                return *this;

        // The buffer must be destroyed and remade either if the buffer size is different
        // or the data is in different places.
        if (m_size != other.m_size || m_device != other.m_device) {
                // If size doesn't match, overwrite m_size and delete old buffer.
                this->~buf();
                m_size = other.m_size;

                if (!m_device)
                        m_data = new float[m_size];
#ifdef BUILD_CUDA_SUPPORT
                else
                        CUDA_CHECK_ERROR(cudaMallocAsync(&m_data, m_size * sizeof(float), m_stream),
                                 "Failed to allocate memory on the GPU.");
#endif // BUILD_CUDA_SUPPORT
        }

        if (!m_device)
                std::memcpy(m_data, other.m_data, m_size * sizeof(float));
#ifdef BUILD_CUDA_SUPPORT
        else
                CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, m_stream), "Failed to copy data in the GPU.");
#endif // BUILD_CUDA_SUPPORT

        return *this;
}

buf &buf::operator= (buf &&other) noexcept
{
        if (this == &other)
                return *this;

        this->~buf();

        m_size = other.m_size;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;

        return *this;
}

bool buf::operator== (const buf &other) const
{
        // Check if sizes match, if not don't check individual values
        if (m_size != other.m_size)
                return false;

        bool ans;
        math::compare(m_size, *this, other, &ans);
        return ans;
}

void buf::move([[maybe_unused]] loc_type location)
{
#ifdef BUILD_CUDA_SUPPORT
        if (location == (m_device ? DEVICE : HOST))
                return;

        float *data;
        if (m_device) {
                data = new float[m_size];
                CUDA_CHECK_ERROR(cudaMemcpyAsync(data, m_data, m_size * sizeof(float),
                        cudaMemcpyDeviceToHost, m_stream), "Failed to copy data from the GPU.");
                CUDA_CHECK_ERROR(cudaFreeAsync(m_data, m_stream), "Failed to free GPU memory.");
        } else {
                CUDA_CHECK_ERROR(cudaMallocAsync(&data, m_size * sizeof(float), m_stream),
                        "Failed to allocate memory on the GPU.");
                CUDA_CHECK_ERROR(cudaMemcpyAsync(data, m_data, m_size * sizeof(float),
                        cudaMemcpyHostToDevice, m_stream), "Failed to copy data to the GPU.");
                delete [] m_data;
        }

        m_device = location == DEVICE;
        m_data   = data;
#endif // BUILD_CUDA_SUPPORT
}

} // namespace nn
