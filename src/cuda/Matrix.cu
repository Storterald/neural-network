#include "Matrix.cuh"

#include "Kernels.cuh"

Matrix::Matrix(
        uint32_t width,
        uint32_t height
)  :
        m_width(width),
        m_height(height),
        m_size(width * height)
{
        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate GPU memory.");
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_data, 0, m_size * sizeof(float)),
                "Failed initialize GPU memory.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::Matrix.");
}

Matrix::Matrix(
        uint32_t width,
        uint32_t height,
        const float *data
)  :
        m_width(width),
        m_height(height),
        m_size(width * height)
{
        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate GPU memory.");
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, data, m_size * sizeof(float), cudaMemcpyHostToDevice),
                "Failed to copy data to GPU.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing Matrix::Matrix.");
}

Matrix::Matrix(const Matrix &other) :
        m_width(other.m_width),
        m_height(other.m_height),
        m_size(other.m_size)
{
        CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                "Failed to allocate GPU memory.");
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float), cudaMemcpyDeviceToDevice),
                "Failed to copy data to GPU.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::Matrix.");
}

Matrix::Matrix(Matrix &&other) noexcept  :
        m_width(other.m_width),
        m_height(other.m_height),
        m_size(other.m_size),
        m_data(other.m_data)
{
        other.m_width = 0;
        other.m_height = 0;
        other.m_size = 0;
        other.m_data = nullptr;

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::Matrix.");
}

Matrix::~Matrix()
{
        CUDA_CHECK_ERROR(cudaFree(m_data),
                "Failed to free GPU memory.");

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::~Matrix.");
}

Matrix &Matrix::operator= (
        const Matrix &other
) {
        if (this == &other)
                return *this;

        if (m_size != other.m_size) {
                m_width = other.m_width;
                m_height = other.m_height;
                m_size = other.m_size;

                CUDA_CHECK_ERROR(cudaFree(m_data),
                        "Failed to free GPU memory.");
                CUDA_CHECK_ERROR(cudaMalloc(&m_data, m_size * sizeof(float)),
                        "Failed to allocate GPU memory.");
        }

        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_data, other.m_data, m_size * sizeof(float), cudaMemcpyDeviceToDevice),
                "Failed to copy vector data on GPU.");

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator=.");
        return *this;
}

Matrix &Matrix::operator= (
        Matrix &&other
) noexcept {
        if (this == &other)
                return *this;

        if (m_data)
                cudaFree(m_data);

        m_data = other.m_data;
        m_size = other.m_size;
        m_width = other.m_width;
        m_height = other.m_height;

        other.m_size = 0;
        other.m_width = 0;
        other.m_height = 0;
        other.m_data = nullptr;

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator=.");
        return *this;
}

void Matrix::operator+= (
        const Matrix &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error("Matrix::operator+=: width or height mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::addKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "addKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator+=.");
}

void Matrix::operator-= (
        const Matrix &other
) {
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw Logger::fatal_error("Matrix::operator-=: width or height mismatch.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::subKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, other.m_data, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "subKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator-=.");
}

Vector Matrix::operator* (
        const Vector &vec
) const {
#ifdef DEBUG_MODE_ENABLED
        if (vec.m_size != m_width)
                throw Logger::fatal_error("Matrix::operator*: Vector size not equal to Matrix width.");
#endif // DEBUG_MODE_ENABLED
        const uint32_t BLOCKS_COUNT { (m_height + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(m_height);
        Kernels::vecMulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.m_data, m_data, vec.m_data, m_width, m_height);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "vecMulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator*.");
        return result;
}

Matrix Matrix::operator* (
        float scalar
) const {
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Matrix result(m_width, m_height);
        Kernels::scalarMulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.m_data, m_data, scalar, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "scalarMulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator*.");
        return result;
}

void Matrix::operator*= (
        float scalar
) {
        const uint32_t BLOCKS_COUNT { (m_size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::scalarMulKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(m_data, m_data, scalar, m_size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "scalarMulKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Matrix::operator*=.");
}
