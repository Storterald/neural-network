#include "Utils.cuh"

#include "Kernels.cuh"
#include "../Base.h"

void Utils::forward(
        float *result,
        const float *input,
        const float *w,
        const float *d,
        uint32_t width,
        uint32_t height
) {
        const uint32_t BLOCKS_COUNT { (height + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::forwardKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(
                result, input, w, d, width, height
        );

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "forwardKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Utils::forward.");
}

void Utils::backward(
        float *previousCosts,
        float *dw,
        const float *db,
        const float *w,
        const float *input,
        uint32_t width,
        uint32_t height
) {
        const uint32_t BLOCKS_COUNT { (width + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::backwardKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(
                previousCosts, dw, db, w, input, width, height
        );

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "backwardKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Utils::backward.");
}

void Utils::fmaScalar(
        float *result,
        const float *a,
        float scalar,
        const float *c,
        uint32_t size
) {
        const uint32_t BLOCKS_COUNT { (size + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Kernels::fmaScalarKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result, a, scalar, c, size);

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "fmaScalarKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Utils::fmaScalar.");
}