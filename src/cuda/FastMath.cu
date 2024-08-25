#include "FastMath.cuh"

#include "Kernels.cuh"

Vector Fast::relu(
        const Vector &vec
) {
        const uint32_t BLOCKS_COUNT { (vec.size() + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(vec.size());
        Kernels::reluKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.data(), vec.data(), vec.size());

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "reluKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Fast::relu.");
        return result;
}

Vector Fast::reluDerivative(
        const Vector &vec
) {
        const uint32_t BLOCKS_COUNT { (vec.size() + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(vec.size());
        Kernels::reluDerivativeKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.data(), vec.data(), vec.size());

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "reluKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Fast::relu'.");
        return result;
}

Vector Fast::tanh(
        const Vector &vec
) {
        const uint32_t BLOCKS_COUNT { (vec.size() + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(vec.size());
        Kernels::tanhKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.data(), vec.data(), vec.size());

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "reluKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Fast::tanh.");
        return result;
}

Vector Fast::tanhDerivative(
        const Vector &vec
) {
        const uint32_t BLOCKS_COUNT { (vec.size() + BLOCK_SIZE - 1) >> BLOCK_BITSHIFT };

        Vector result(vec.size());
        Kernels::tanhDerivativeKernel<<<BLOCKS_COUNT, BLOCK_SIZE>>>(result.data(), vec.data(), vec.size());

#ifdef DEBUG_MODE_ENABLED
        checkCudaError(cudaGetLastError(), "reluKernel launch failed.");
#endif // DEBUG_MODE_ENABLED

        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in Fast::tanh'.");
        return result;
}