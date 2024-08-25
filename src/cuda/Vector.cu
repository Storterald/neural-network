#include "Vector.cuh"

#include "Kernels.cuh"

Vector::Vector(
        uint32_t size
) : m_size(size) {
        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate GPU memory.");
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_data, 0, m_size * sizeof(float)),
                "Failed initialize GPU memory.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::Vector.");
}

Vector::Vector(
        uint32_t size,
        const float *data
) : m_size(size) {
        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate GPU memory.");
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, data, m_size * sizeof(float), cudaMemcpyHostToDevice),
                "Failed to copy data to GPU.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::Vector.");
}

Vector::Vector(const Vector &other) : m_size(other.m_size) {
        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate GPU memory.");
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float), cudaMemcpyDeviceToDevice),
                "Failed to copy data to GPU.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::Vector.");
}

Vector::Vector(Vector &&other) noexcept :
        m_size(other.m_size),
        m_data(other.m_data)
{
        other.m_size = 0;
        other.m_data = nullptr;

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::Vector.");
}

Vector::~Vector()
{
        CUDA_CHECK_ERROR(cudaFree(m_data),
                "Failed to free GPU memory.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::~Vector.");
}

Vector &Vector::operator= (
        const Vector &other
) {
        if (this == &other)
                return *this;

        if (m_size != other.m_size) {
                m_size = other.m_size;

                CUDA_CHECK_ERROR(cudaFree(m_data),
                        "Failed to free GPU memory.");
                CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                        "Failed to allocate GPU memory.");
        }

        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float), cudaMemcpyDeviceToDevice),
                "Failed to copy vector data on GPU.");

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator=.");
        return *this;
}

Vector &Vector::operator= (
        Vector &&other
) noexcept {
        if (this == &other)
                return *this;

        cudaFree(m_data);

        m_size = other.m_size;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator=.");
        return *this;
}

Vector Vector::operator+ (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector::operator+: Size mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(m_size);
        Kernels::addKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "addKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator+.");
        return result;
}

Vector Vector::operator- (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector::operator-: Size mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(m_size);
        Kernels::subKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "subKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator-.");
        return result;
}

Vector Vector::operator* (
        const Vector &other
) const {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector::operator*: Size mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(m_size);
        Kernels::mulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "mulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator*.");
        return result;
}

void Vector::operator+= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector::operator+=: Size mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::addKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "addKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator+=.");
}

void Vector::operator-= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector::operator-=: Size mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::subKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "subKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator-=.");
}

void Vector::operator*= (
        const Vector &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_size != other.m_size)
                throw Logger::fatal_error("Vector::operator*=: Size mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::mulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "mulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator*=.");
}

void Vector::operator*= (
        float scalar
) {
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::scalarMulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, scalar, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "scalarMulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator*=.");
}

Vector Vector::operator* (
        float scalar
) const {
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(m_size);
        Kernels::scalarMulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.m_data, m_data, scalar, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "scalarMulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Vector::operator*.");
        return result;
}