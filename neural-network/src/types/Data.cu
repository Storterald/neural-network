#include "Data.h"

namespace Kernels {

        __global__ void compare(uint32_t size, const float first[], const float second[], bool *ans)
        {
                *ans = true;
                for (uint32_t i = 0; i < size && *ans; ++i)
                        *ans  = first[i] == second[i];
        }

} // namespace Kernels

Data::Data(uint32_t size) : m_size(size), m_device(size >= CUDA_MINIMUM)
{
        if (!m_device) {
                m_data = new float[m_size]();
                return;
        }

        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_data, 0, m_size * sizeof(float), 0),
                "Failed to set memory in the GPU.");
        CUDA_CHECK_ERROR(cudaStreamSynchronize(0), "Error synchronizing in Data::Data");
}

Data::Data(const Data &other) : m_size(other.m_size), m_device(other.m_device)
{
        if (!m_device) {
                m_data = new float[m_size];
                std::memcpy(m_data, other.m_data, m_size * sizeof(float));
                return;
        }

        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate memory on the GPU.");
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float),
                cudaMemcpyDeviceToDevice, 0), "Failed to copy data in the GPU.");
        CUDA_CHECK_ERROR(cudaStreamSynchronize(0), "Error synchronizing in Data::Data");
}

Data::Data(Data &&other) noexcept : m_size(other.m_size), m_data(other.m_data)
{

        other.m_size = 0;
        other.m_data = nullptr;
}

Data::~Data()
{
        if (!m_data)
                return;

        if (!m_device)
                delete [] m_data;
        else
                CUDA_CHECK_ERROR(cudaFree(m_data), "Failed to free GPU memory.");
}

Data &Data::operator= (const Data &other)
{
        if (this == &other)
                return *this;

        // The buffer must be destroyed and remade either if the buffer size is different
        // or the data is in different places.
        if (m_size != other.m_size || m_device != other.m_device) {
                // If size doesn't match, overwrite m_size and delete old buffer.
                this->~Data();
                m_size = other.m_size;

                if (!m_device)
                        m_data = new float[m_size];
                else
                        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                                 "Failed to allocate memory on the GPU.");
        }

        if (!m_device) {
                std::memcpy(m_data, other.m_data, m_size * sizeof(float));
        } else {
                CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, 0), "Failed to copy data in the GPU.");
                CUDA_CHECK_ERROR(cudaStreamSynchronize(0), "Error synchronizing in Data::operator=");
        }

        return *this;
}

Data &Data::operator= (Data &&other) noexcept
{
        if (this == &other)
                return *this;

        this->~Data();

        m_size = other.m_size;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;

        return *this;
}

bool Data::operator== (const Data &other) const
{
        // Check if sizes match, if not don't check individual values
        if (m_size != other.m_size)
                return false;

        bool ans = true;
        if (!m_device) {
                for (uint32_t i = 0; i < m_size && ans; ++i)
                        ans = m_data[i] == other.m_data[i];

                return ans;
        }

        bool *d_ans;
        CUDA_CHECK_ERROR(cudaMalloc(&d_ans, sizeof(bool)),
                "Failed to allocate GPU memory.");

        const uint32_t BLOCKS_COUNT = (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::compare<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_size, m_data, other.m_data, d_ans);

        CUDA_CHECK_ERROR(cudaMemcpy(&ans, d_ans, sizeof(bool),
                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");
        CUDA_CHECK_ERROR(cudaFree(d_ans), "Failed to free GPU memory.");
        return ans;
}
