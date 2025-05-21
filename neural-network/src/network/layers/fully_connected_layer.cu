#include <neural-network/network/layers/fully_connected_layer.h>

#include <host_defines.h> // __global__

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

                const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

                if (k >= width)
                        return;

                float dCe = 0.0f;
                for (uint32_t j = 0; j < height; j++) {
                        dCe += w[j * width + k] * db[j];
                        dw[j * width + k] = input[k] * db[j];
                }

                result[k] = dCe;
        }

} // namespace kernels

namespace nn {

void fully_connected_layer::_d_backward(
        const float        input[],
        float              dw[],
        const float        db[],
        float              result[]) const {

        const uint32_t BLOCKS_COUNT = (m_w.width() + CUDA_THREADS - 1) / CUDA_THREADS;
        kernels::backward<<<BLOCKS_COUNT, CUDA_THREADS>>>(
                m_w.width(), m_w.height(), input,
                m_w.view(nn::buf::DEVICE), dw, db, result);

        cuda::check_last_error("backward kernel launch failed.");
}

} // namespace nn
