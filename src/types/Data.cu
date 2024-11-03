#include "Data.h"

#include "Base.h"

namespace Kernels {

        __global__ void compare(uint32_t size, const float first[], const float second[], bool *ans)
        {
                // Reset d_compareAns
                *ans = true;

                // Stop as soon as values differ
                for (uint32_t i{}; i < size && *ans; ++i)
                        *ans  = first[i] == second[i];
        }

} // namespace Kernels

Data::Data(uint32_t size) :
        m_size(size), m_data(nullptr)
{
        // Size must be higher than 0 for cudaMallocManaged to work.
        if (m_size == 0)
                return;

        // Unified memory is accessible from both CPU and GPU
        CUDA_CHECK_ERROR(cudaMallocManaged(&m_data, m_size * sizeof(float)),
                "Failed to allocate unified memory.");
}

Data::Data(const Data &other) :
        m_size(other.m_size), m_data(nullptr)
{
        // Don't alloc nor copy if the size is 0.
        if (m_size == 0)
                return;

        CUDA_CHECK_ERROR(cudaMallocManaged(&m_data, m_size * sizeof(float)),
                "Failed to allocate unified memory.");

        // Unified memory is allocated on the CPU by default, so a simple std::memcpy
        // should be faster.
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));
}

Data::Data(Data &&other) noexcept :
        m_size(other.m_size), m_data(other.m_data)
{
        other.m_size = 0;
        other.m_data = nullptr;
}

Data::~Data()
{
        if (!m_data)
                return;

        CUDA_CHECK_ERROR(cudaFree(m_data),
                "Failed to free GPU memory.");
}

Data &Data::operator= (
        const Data &other
) {
        if (this == &other)
                return *this;

        // The buffer must be destroyed and remade either if the buffer size is different
        // or the data is in different places.
        if (m_size != other.m_size) {
                // If size doesn't match, overwrite m_size and delete old buffer.
                m_size = other.m_size;
                if (m_data)
                        CUDA_CHECK_ERROR(cudaFree(m_data),
                                "Failed to free GPU memory.");

                CUDA_CHECK_ERROR(cudaMallocManaged(&m_data, m_size * sizeof(float)),
                        "Failed to allocate unified memory.");
        }

        // Once the buffer has the correct size, the data can be copied.
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));

        return *this;
}

Data &Data::operator= (
        Data &&other
) noexcept {
        if (this == &other)
                return *this;

        if (m_data)
                CUDA_CHECK_ERROR(cudaFree(m_data),
                        "Failed to free GPU memory.");

        m_size = other.m_size;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;

        return *this;
}

bool Data::operator== (
        const Data &other
) const {
        // Check if sizes match, if not don't check individual values
        if (m_size != other.m_size)
                return false;

        bool ans { true };
        if (m_size > CUDA_MINIMUM) {
                for (uint32_t i{}; i < m_size && ans; ++i)
                        ans = m_data[i] == other.m_data[i];
        } else {
                // Allocate space for the kernel result.
                bool *d_ans;
                CUDA_CHECK_ERROR(cudaMalloc(&d_ans, sizeof(bool)),
                        "Failed to allocate GPU memory.");

                const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };
                Kernels::compare<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_size, m_data, other.m_data, d_ans);

                // Copy the result to CPU memory, the default value 'true' is discarded
                // and overwritten with the correct answer.
                CUDA_CHECK_ERROR(cudaMemcpy(&ans, d_ans, sizeof(bool), cudaMemcpyDeviceToHost),
                        "Failed to copy data from the GPU.");

                // Free the kernel answer
                CUDA_CHECK_ERROR(cudaFree(d_ans),
                        "Failed to free GPU memory.");
        }

        // The warning is caused by cudaMalloc and cudaFree, the malloc is counted
        // but the free is not, this causes CLion to emit a warning.
        return ans;
}

