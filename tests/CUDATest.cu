#include "CUDATest.h"

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

}

void CUDATest::access_values(const Data &data)
{
        Kernels::access_values<<<1, 1>>>(data.size(), data.data());

        CUDA_CHECK_ERROR(cudaGetLastError(),
                "getPtrFromData kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in CUDATest::getPtrFromData.");
}

bool CUDATest::check_values(const Data &data, float v)
{
        bool *d_res{};
        CUDA_CHECK_ERROR(cudaMalloc(&d_res, sizeof(bool)),
                "Failed to allocate memory on GPU.");

        Kernels::check_values<<<1, 1>>>(data.size(), data.data(), v, d_res);

        CUDA_CHECK_ERROR(cudaGetLastError(),
                "checkValues kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in CUDATest::checkValues.");

        bool res{};
        CUDA_CHECK_ERROR(cudaMemcpy(&res, d_res, sizeof(bool), cudaMemcpyDeviceToHost),
                "Failed to copy data from the GPU.");

        CUDA_CHECK_ERROR(cudaFree(d_res),
                "Failed to free GPU memory.");
        return res;
}