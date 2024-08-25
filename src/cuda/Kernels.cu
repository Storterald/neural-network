#include "Kernels.cuh"

#include "FastMath.cuh"

namespace Fast {

        // std::max is not usable from kernels as it is __host__ function.
        __device__ inline float max(float a, float b)
        {
                return a > b ? a : b;
        }

        __device__ inline float tanh(float x)
        {
                // Around 6-8 times faster than std::tanh in release mode

                // For |x| > 4.9, tanh(x) is approximately 1 or -1. Returns at 4.9
                // since it's the highest decimal the output isn't higher than 1
                if (std::abs(x) >= 4.9f)
                        return std::copysign(1.0f, x);

                // 7th-order approximation (accurate within 0.000085).
                // https://www.desmos.com/calculator/2myik1oe4x
                const float x2 { x * x };
                return x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) / (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }

        __device__ inline float tanhDerivative(float x)
        {
                // Uses the tanhFast to calculate the derivative way faster than
                // with using std::tanh, also more accurate than tanhFast.

                // tanh'(0) = 1
                if (x == 0.0f)
                        return 1.0f;

                // After this point, the approximation of the derivative goes negative.
                // Instead, the tanh' goes closer and closer to 0 (accurate within 0.000170).
                // Returns at 4.9 since it's the highest decimal the value is positive.
                if (std::abs(x) > 4.9f)
                        return 0.0f;

                const float tanh { Fast::tanh(x) };
                return 1.0f - tanh * tanh;
        }

} // namespace Fast

__global__ void Kernels::addKernel(
        float *const result,
        const float *a,
        const float *b,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = a[idx] + b[idx];
}

__global__ void Kernels::subKernel(
        float *const result,
        const float *a,
        const float *b,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = a[idx] - b[idx];
}

__global__ void Kernels::mulKernel(
        float *const result,
        const float *a,
        const float *b,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = a[idx] * b[idx];
}

__global__ void Kernels::scalarMulKernel(
        float *const result,
        const float *a,
        float scalar,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = a[idx] * scalar;
}

__global__ void Kernels::vecMulKernel(
        float *const result,
        const float *matrix,
        const float *vector,
        uint32_t width,
        uint32_t height
) {
        const uint32_t row { blockIdx.x * blockDim.x + threadIdx.x };
        if (row >= height)
                return;

        float sum = 0.0f;
        for (uint32_t col { 0 }; col < width; col++)
                sum += matrix[row * width + col] * vector[col];

        result[row] = sum;
}

__global__ void Kernels::fmaScalarKernel(
        float *result,
        const float *a,
        float scalar,
        const float *c,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx >= size)
                return;

        result[idx] = a[idx] * scalar + c[idx];
}

__global__ void Kernels::reluKernel(
        float *const result,
        const float *data,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = Fast::max(0.0f, data[idx]);
}

__global__ void Kernels::reluDerivativeKernel(
        float *const result,
        const float *data,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = data[idx] >= 0.0f ? 1.0f : 0.0f;
}

__global__ void Kernels::tanhKernel(
        float *const result,
        const float *data,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = Fast::tanh(data[idx]);
}

__global__ void Kernels::tanhDerivativeKernel(
        float *const result,
        const float *data,
        uint32_t size
) {
        const uint32_t idx { blockIdx.x * blockDim.x + threadIdx.x };
        if (idx < size)
                result[idx] = Fast::tanhDerivative(data[idx]);
}

__global__ void Kernels::forwardKernel(
        float *result,
        const float *input,
        const float *w,
        const float *d,
        uint32_t width,
        uint32_t height
) {
        const uint32_t row { blockIdx.x * blockDim.x + threadIdx.x };
        if (row >= height)
                return;

        float sum = 0.0f;
        for (uint32_t col { 0 }; col < width; col++)
                sum += w[row * width + col] * input[col];

        result[row] = sum + d[row];
}

__global__ void Kernels::backwardKernel(
        float *previousCosts,
        float *dw,
        const float *db,
        const float *w,
        const float *input,
        uint32_t width,
        uint32_t height
) {
        const uint32_t k { blockIdx.x * blockDim.x + threadIdx.x };

        if (k >= width)
                return;

        float dCe { 0.0f };
        for (uint32_t j { 0 }; j < height; j++) {
                dCe += w[j * width + k] * db[j];
                dw[j * width + k] = input[k] * db[j];
        }

        previousCosts[k] = dCe;
}