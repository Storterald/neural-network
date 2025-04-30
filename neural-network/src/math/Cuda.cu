#include "_math.h"

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cmath>
#include <cuda_runtime.h>

#include <cstdint>

#include <neural-network/utils/macros.h>
#include <neural-network/types/memory.h>

namespace utils {

        __device__ inline float tanh(float x)
        {
                if (cuda::std::fabsf(x) >= 4.9f)
                        return cuda::std::copysignf(1.0f, x);

                const float x2 = x * x;
                return x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) /
                       (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }

        __device__ inline float tanh_derivative(float x)
        {
                if (x == 0.0f)
                        return 1.0f;

                if (cuda::std::fabsf(x) > 4.9f)
                        return 0.0f;

                const float tanh = utils::tanh(x);
                return 1.0f - tanh * tanh;
        }

} // namespace utils

namespace kernels {

        __global__ void sum(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < size)
                        result[idx] = first[idx] + second[idx];
        }

        __global__ void sub(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < size)
                        result[idx] = first[idx] - second[idx];
        }

        __global__ void mul(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] * second[idx];
        }

        __global__ void div(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = first[idx] / second[idx];
        }

        __global__ void sum(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = data[idx] + scalar;
        }

        __global__ void sub(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = data[idx] - scalar;
        }

        __global__ void mul(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = data[idx] * scalar;
        }

        __global__ void div(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = data[idx] / scalar;
        }

        __global__ void tanh(
                uint32_t           size,
                const float        data[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = utils::tanh(data[idx]);
        }

        __global__ void tanh_derivative(
                uint32_t           size,
                const float        data[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = utils::tanh_derivative(data[idx]);
        }

        __global__ void ReLU(
                uint32_t           size,
                const float        data[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = cuda::std::max(0.0f, data[idx]);
        }

        __global__ void ReLU_derivative(
                uint32_t           size,
                const float        data[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = data[idx] >= 0.0f ? 1.0f : 0.0f;
        }

        __global__ void min(
                uint32_t           size,
                const float        data[],
                float              min,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = cuda::std::min(data[idx], min);
        }

        __global__ void max(
                uint32_t           size,
                const float        data[],
                float              max,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = cuda::std::max(data[idx], max);
        }

        __global__ void clamp(
                uint32_t           size,
                const float        data[],
                float              min,
                float              max,
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < size)
                        result[idx] = cuda::std::clamp(data[idx], min, max);
        }

        __global__ void min(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = cuda::std::min(first[idx], second[idx]);
        }

        __global__ void max(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = cuda::std::max(first[idx], second[idx]);
        }

        __global__ void clamp(
                uint32_t           size,
                const float        data[],
                const float        min[],
                const float        max[],
                float              result[]) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                if (idx < size)
                        result[idx] = cuda::std::clamp(data[idx], min[idx], max[idx]);
        }

        __global__ void compare(
                uint32_t           size,
                const float        first[],
                const float        second[],
                bool               *result) {

                const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

                if (idx < size && first[idx] != second[idx])
                        *result = false;
        }

        __global__ void matvec_mul(
                uint32_t           width,
                uint32_t           height,
                const float        matrix[],
                const float        vector[],
                float              result[]) {

                const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
                if (row >= height)
                        return;

                float sum = 0.0f;
                for (uint32_t col = 0; col < width; col++)
                        sum += matrix[row * width + col] * vector[col];

                result[row] = sum;
        }

} // namespace kernels

#define DECLARE_CUDA_FUNCTION(__name__, __size__, ...)                                                          \
template<> void _math<MATH_CUDA>:: __name__ (GET_ARGS(__VA_ARGS__))                                             \
{                                                                                                               \
        const uint32_t BLOCKS_COUNT = ((__size__) + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;                          \
        kernels:: __name__ <<<BLOCKS_COUNT, BLOCK_SIZE>>>(GET_ARGS_NAMES(__VA_ARGS__));                         \
        CUDA_CHECK_ERROR(cudaGetLastError(), "kernels::" #__name__ " launch failed.");                          \
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in _math<MATH_CUDA>::" #__name__);       \
}

namespace nn {

DECLARE_CUDA_FUNCTION(sum, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CUDA_FUNCTION(sub, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CUDA_FUNCTION(mul, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CUDA_FUNCTION(div, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CUDA_FUNCTION(sum, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CUDA_FUNCTION(sub, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CUDA_FUNCTION(mul, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CUDA_FUNCTION(div, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CUDA_FUNCTION(tanh, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CUDA_FUNCTION(tanh_derivative, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CUDA_FUNCTION(ReLU, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CUDA_FUNCTION(ReLU_derivative, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CUDA_FUNCTION(min, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CUDA_FUNCTION(max, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CUDA_FUNCTION(clamp, size,
        uint32_t,             size,
        const float *,        data,
        const float *,        min,
        const float *,        max,
        float *,              result)

DECLARE_CUDA_FUNCTION(min, size,
        uint32_t,             size,
        const float *,        data,
        float,                min,
        float *,              result)

DECLARE_CUDA_FUNCTION(max, size,
        uint32_t,             size,
        const float *,        data,
        float,                max,
        float *,              result)

DECLARE_CUDA_FUNCTION(clamp, size,
        uint32_t,             size,
        const float *,        data,
        float,                min,
        float,                max,
        float *,              result)

template<> void _math<MATH_CUDA>::compare(
        uint32_t           size,
        const float        first[],
        const float        second[],
        bool               *result) {

        *result = true;
        span s(1, true, result, false, true);

        const uint32_t BLOCKS_COUNT = (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT;
        kernels::compare<<<BLOCKS_COUNT, BLOCK_SIZE>>>(size, first, second, s);
        CUDA_CHECK_ERROR(cudaGetLastError(), "Kernels::compare launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in _Math<MATH_CUDA>::compare");
}

DECLARE_CUDA_FUNCTION(matvec_mul, width,
        uint32_t,             width,
        uint32_t,             height,
        const float *,        matrix,
        const float *,        vector,
        float *,              result)

} // namespace nn
