#include "CudaTestHelper.h"

namespace Kernels {
        
        __global__ void access_values(uint32_t size, const float *data)
        {
                for (uint32_t i = 0; i < size; ++i)
                        volatile float value = data[i];
        }

        __global__ void check_values(uint32_t size, const float *data, float v, bool *res)
        {
                *res = true;
                for (uint32_t i = 0; i < size; ++i)
                        *res &= (data[i] == v);
        }

} // namespace Kernels

void Helper::access_values(uint32_t size, const float *data)
{
        Kernels::access_values<<<1, 1>>>(size, data);

        CUDA_CHECK_ERROR(cudaGetLastError(), "Kernels::access_values launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in Helper::access_values.");
}

bool Helper::check_values(uint32_t size, const float *data, float v)
{
        bool *d_res = nullptr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_res, sizeof(bool)),
                "Failed to allocate memory on GPU.");

        Kernels::check_values<<<1, 1>>>(size, data, v, d_res);

        CUDA_CHECK_ERROR(cudaGetLastError(), "Kernels::check_values launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in Helper::check_values.");

        bool res = false;
        CUDA_CHECK_ERROR(cudaMemcpy(&res, d_res, sizeof(bool), cudaMemcpyDeviceToHost),
                "Failed to copy data from the GPU.");

        CUDA_CHECK_ERROR(cudaFree(d_res), "Failed to free GPU memory.");
        return res;
}
