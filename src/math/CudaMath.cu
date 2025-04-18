#include "Math.h"

#include "_Math.h"
#include "Base.h"

namespace Utils {

        __device__ inline float min(float a, float b)
        {
                return a < b ? a : b;
        }

        __device__ inline float max(float a, float b)
        {
                return a > b ? a : b;
        }

        __device__ inline float tanh(float x)
        {
                if (fabsf(x) >= 4.9f)
                        return copysignf(1.0f, x);

                const float x2 = x * x;
                return x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) /
                       (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }

        __device__ inline float tanhDerivative(float x)
        {
                if (x == 0.0f)
                        return 1.0f;

                if (fabsf(x) > 4.9f)
                        return 0.0f;

                const float tanh = Utils::tanh(x);
                return 1.0f - tanh * tanh;
        }

}

namespace Kernels {

        __global__ void sum(uint32_t size, const float first[], const float second[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < size)
                        result[idx] = first[idx] + second[idx];
        }

        __global__ void sub(uint32_t size, const float first[], const float second[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < size)
                        result[idx] = first[idx] - second[idx];
        }

        __global__ void mul(uint32_t size, const float first[], const float second[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] * second[idx];
        }

        __global__ void div(uint32_t size, const float first[], const float second[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] / second[idx];
        }

        __global__ void sum(uint32_t size, const float first[], float scalar, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] + scalar;
        }

        __global__ void sub(uint32_t size, const float first[], float scalar, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] - scalar;
        }

        __global__ void mul(uint32_t size, const float first[], float scalar, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] * scalar;
        }

        __global__ void div(uint32_t size, const float first[], float scalar, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] / scalar;
        }

        __global__ void tanh(uint32_t size, const float data[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::tanh(data[idx]);
        }

        __global__ void tanhDerivative(uint32_t size, const float data[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::tanhDerivative(data[idx]);
        }

        __global__ void ReLU(uint32_t size, const float data[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::max(0.0f, data[idx]);
        }

        __global__ void ReLUDerivative(uint32_t size, const float data[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = data[idx] >= 0.0f ? 1.0f : 0.0f;
        }

        __global__ void min(uint32_t size, const float a[], float min, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::min(a[idx], min);
        }

        __global__ void max(uint32_t size, const float a[], float max, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::max(a[idx], max);
        }

        __global__ void clamp(uint32_t size, const float data[], float min, float max, float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::min(Utils::max(data[idx], min), max);
        }

        __global__ void min(uint32_t size, const float a[], const float b[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::min(a[idx], b[idx]);
        }

        __global__ void max(uint32_t size, const float a[], const float b[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::max(a[idx], b[idx]);
        }

        __global__ void clamp(uint32_t size, const float data[], const float min[], const float max[], float result[])
        {
                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = Utils::min(Utils::max(data[idx], min[idx]), max[idx]);
        }

        __global__ void matrixVectorMul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[])
        {
                const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
                if (row >= height)
                        return;

                float sum = 0.0f;
                for (uint32_t col = 0; col < width; col++)
                        sum += matrix[row * width + col] * vector[col];

                result[row] = sum;
        }

} // namespace Kernels

template<> void _Math<MATH_CUDA>::sum(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::sum<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "sum kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::sum.");
}

template<> void _Math<MATH_CUDA>::sub(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::sub<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "sub kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::sub.");
}

template<> void _Math<MATH_CUDA>::mul(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::mul<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "mul kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::mul.");
}

template<> void _Math<MATH_CUDA>::div(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::div<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "div kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::div.");
}

template<> void _Math<MATH_CUDA>::sum(uint32_t size, const float data[], float scalar, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::sum<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, scalar, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "sum kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::sum.");
}

template<> void _Math<MATH_CUDA>::sub(uint32_t size, const float data[], float scalar, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::sub<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, scalar, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "sub kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::sub.");
}

template<> void _Math<MATH_CUDA>::mul(uint32_t size, const float data[], float scalar, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::mul<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, scalar, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "mul kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::mul.");
}

template<> void _Math<MATH_CUDA>::div(uint32_t size, const float data[], float scalar, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::div<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, scalar, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "div kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::div.");
}

template<> void _Math<MATH_CUDA>::tanh(uint32_t size, const float data[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::tanh<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "tanh kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::tanh.");
}

template<> void _Math<MATH_CUDA>::tanh_derivative(uint32_t size, const float data[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::tanhDerivative<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "tanh_derivative kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::tanh_derivative.");
}

template<> void _Math<MATH_CUDA>::ReLU(uint32_t size, const float data[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::ReLU<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "ReLU kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::ReLU.");
}

template<> void _Math<MATH_CUDA>::ReLU_derivative(uint32_t size, const float data[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::ReLUDerivative<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "ReLU_derivative kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::ReLU_derivative.");
}

template<> void _Math<MATH_CUDA>::min(uint32_t size, const float data[], float min, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::min<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, min, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "min kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::min.");
}

template<> void _Math<MATH_CUDA>::max(uint32_t size, const float data[], float max, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::max<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, max, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "max kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::max.");
}

template<> void _Math<MATH_CUDA>::clamp(uint32_t size, const float data[], float min, float max, float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::clamp<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, min, max, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "clamp kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::clamp.");
}

template<> void _Math<MATH_CUDA>::min(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::min<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "min kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::min.");
}

template<> void _Math<MATH_CUDA>::max(uint32_t size, const float first[], const float second[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::max<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "max kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::max.");
}

template<> void _Math<MATH_CUDA>::clamp(uint32_t size, const float data[], const float min[], const float max[], float result[])
{
        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::clamp<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, data, min, max, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "clamp kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::clamp.");
}

template<> void _Math<MATH_CUDA>::matvec_mul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[])
{
        const uint32_t BLOCKS_COUNT = (width + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        Kernels::matrixVectorMul<<<BLOCKS_COUNT, BLOCK_SIZE>>>(width, height, matrix, vector, result);

        CUDA_CHECK_ERROR(cudaGetLastError(), "matrixVectorMul kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in CudaMath::matrixVectorMul.");
}