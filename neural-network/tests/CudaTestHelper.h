#pragma once

#ifndef BUILD_CUDA_SUPPORT
#error CudaTestHelper.h cannot be included without BUILD_CUDA_SUPPORT
#endif // !BUILD_CUDA_SUPPORT

#include <neural-network/types/buf.h>

namespace helper {

        void access_values(uint32_t size, const float *data);
        bool check_values(uint32_t size, const float *data, float v);
        void set_values(uint32_t size, float *data, float v);

} // namespace helper
