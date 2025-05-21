#include "CudaTestHelper.h"

#include <neural-network/utils/cuda.h>

namespace kernels {
        
        __global__ void access_values(uint32_t size, const float *data)
        {
                for (uint32_t i = 0; i < size; ++i)
                        [[maybe_unused]] volatile float value = data[i];
        }

        __global__ void check_values(uint32_t size, const float *data, float v, bool *res)
        {
                *res = true;
                for (uint32_t i = 0; i < size; ++i)
                        *res &= data[i] == v;
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

        nn::cuda::check_last_error("Kernels::access_values launch failed.");
        nn::cuda::sync_all();
}

bool helper::check_values(uint32_t size, const float *data, float v)
{
        bool *d_res = nn::cuda::alloc<bool>();
        kernels::check_values<<<1, 1>>>(size, data, v, d_res);

        nn::cuda::check_last_error("Kernels::check_values launch failed.");
        nn::cuda::sync_all();

        bool res = false;
        nn::cuda::memcpy(&res, d_res, sizeof(bool), cudaMemcpyDeviceToHost);

        nn::cuda::free(d_res);
        return res;
}

void helper::set_values(uint32_t size, float *data, float v)
{
        kernels::set_values<<<1, 1>>>(size, data, v);

        nn::cuda::check_last_error("Kernels::access_values launch failed.");
        nn::cuda::sync_all();
}
