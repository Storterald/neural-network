#include "CUDATest.h"

#include "../src/math/Base.h"

namespace Kernels {
        
        __global__ void accessValues(uint32_t size, const float *data)
        {
                for (int i{}; i < size; ++i)
                        volatile float value = data[i];
        }

        __global__ void checkValues(uint32_t size, const float *data, float v, bool *res)
        {
                *res = true;
                for (int i{}; i < size; ++i)
                        *res &= (data[i] == v);
        }

}

void CUDATest::accessValues(const Data &data)
{
        Kernels::accessValues<<<1, 1>>>(data.size(), data.data());

        CUDA_CHECK_ERROR(cudaGetLastError(),
                "getPtrFromData kernel launch failed.");
        CUDA_CHECK_ERROR(cudaDeviceSynchronize(),
                "Error synchronizing in CUDATest::getPtrFromData.");
}

bool CUDATest::checkValues(const Data &data, float v)
{
        bool *d_res{};
        CUDA_CHECK_ERROR(cudaMalloc(&d_res, sizeof(bool)),
                "Failed to allocate memory on GPU.");

        Kernels::checkValues<<<1, 1>>>(data.size(), data.data(), v, d_res);

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