#include <neural-network/network/layers/dense_layer.h>

#include <cuda_runtime.h>

#include <cstdint>

#include <neural-network/utils/cuda.h>
#include <neural-network/types/buf.h>
#include <neural-network/cuda_base.h>

namespace kernels {

        __global__ void backward(
                uint32_t           width,
                uint32_t           height,
                const float        input[],
                const float        w[],
                float              dw[],
                const float        db[],
                float              result[]) {

                const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

                if (j >= height)
                        return;

                const uint32_t idx = j * width;
                for (uint32_t k = 0; k < width; ++k) {
                        dw[idx + k] = input[k]           * db[j];
                        atomicAdd(&result[k], w[idx + k] * db[j]);
                }
        }

} // namespace kernels

namespace nn {

void dense_layer::_gpu_backward(
        const float        input[],
        float              dw[],
        const float        db[],
        float              result[]) const {

        const uint32_t BLOCKS_COUNT = (m_w.height() + CUDA_THREADS - 1) / CUDA_THREADS;
        kernels::backward<<<BLOCKS_COUNT, CUDA_THREADS, 0, m_stream>>>(
                m_w.width(), m_w.height(), input,
                m_w.begin().get(), dw, db, result);

        cuda::check_last_error("backward kernel launch failed.");
}

} // namespace nn
