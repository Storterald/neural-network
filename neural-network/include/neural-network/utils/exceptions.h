#pragma once

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaError_t
#endif // BUILD_CUDA_SUPPORT

#include <stdexcept>

namespace nn {

struct fatal_error : std::runtime_error {
        explicit fatal_error(const char *message);
};

#ifdef BUILD_CUDA_SUPPORT
struct cuda_error : fatal_error {
        cuda_error(const char *message, cudaError_t error);
};

struct cuda_bad_alloc : cuda_error {
        explicit cuda_bad_alloc(cudaError_t error);
};
#endif // BUILD_CUDA_SUPPORT

} // namespace nn