#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Kernels {

        __global__ void addKernel(
                float *result,
                const float *a,
                const float *b,
                uint32_t size
        );

        __global__ void subKernel(
                float *result,
                const float *a,
                const float *b,
                uint32_t size
        );

        __global__ void mulKernel(
                float *result,
                const float *a,
                const float *b,
                uint32_t size
        );

        __global__ void scalarMulKernel(
                float *result,
                const float *a,
                float scalar,
                uint32_t size
        );

        __global__ void vecMulKernel(
                float *result,
                const float *matrix,
                const float *vector,
                uint32_t width,
                uint32_t height
        );

        // result[idx] = a[idx] * scalar + c[idx]
        __global__ void fmaScalarKernel(
                float *result,
                const float *a,
                float scalar,
                const float *c,
                uint32_t size
        );

        __global__ void reluKernel(
                float *result,
                const float *data,
                uint32_t size
        );

        __global__ void reluDerivativeKernel(
                float *result,
                const float *data,
                uint32_t size
        );

        __global__ void tanhKernel(
                float *result,
                const float *data,
                uint32_t size
        );

        __global__ void tanhDerivativeKernel(
                float *result,
                const float *data,
                uint32_t size
        );

        // BLOCKS_COUNT should be calculated using w.height()
        __global__ void forwardKernel(
                float *result,
                const float *input,
                const float *w,
                const float *d,
                uint32_t width,
                uint32_t height
        );

        // BLOCKS_COUNT should be calculated using w.width()
        __global__ void backwardKernel(
                float *previousCosts,
                float *dw,
                const float *db,
                const float *w,
                const float *input,
                uint32_t width,
                uint32_t height
        );

} // namespace Kernels