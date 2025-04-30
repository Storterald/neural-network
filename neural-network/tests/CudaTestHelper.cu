#include "CudaTestHelper.h"

namespace kernels {
        
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

        __global__ void set_values(uint32_t size, float *data, float v)
        {
                for (uint32_t i = 0; i < size; ++i)
                        data[i] = v;
        }

} // namespace kernels

void helper::access_values(uint32_t size, const float *data)
{
        kernels::access_values<<<1, 1>>>(size, data);

        CUDA_CHECK_ERROR(cudaGetLastError(), "Kernels::access_values launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in helper::access_values.");
}

bool helper::check_values(uint32_t size, const float *data, float v)
{
        bool *d_res = nullptr;
        CUDA_CHECK_ERROR(cudaMalloc(&d_res, sizeof(bool)),
                "Failed to allocate memory on GPU.");

        kernels::check_values<<<1, 1>>>(size, data, v, d_res);

        CUDA_CHECK_ERROR(cudaGetLastError(), "Kernels::check_values launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in helper::check_values.");

        bool res = false;
        CUDA_CHECK_ERROR(cudaMemcpy(&res, d_res, sizeof(bool), cudaMemcpyDeviceToHost),
                "Failed to copy data from the GPU.");

        CUDA_CHECK_ERROR(cudaFree(d_res), "Failed to free GPU memory.");
        return res;
}

void helper::set_values(uint32_t size, float *data, float v)
{
        kernels::set_values<<<1, 1>>>(size, data, v);

        CUDA_CHECK_ERROR(cudaGetLastError(), "kernels::access_values launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "Error synchronizing in helper::access_values.");
}
