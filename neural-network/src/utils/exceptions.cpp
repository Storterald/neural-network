#include <neural-network/utils/exceptions.h>

#ifdef BUILD_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <driver_types.h> // cudaError_t
#endif // BUILD_CUDA_SUPPORT

#include <stdexcept>
#include <iostream>

#ifdef BUILD_CUDA_SUPPORT
#include <format>
#endif // BUILD_CUDA_SUPPORT

#include <neural-network/utils/logger.h>

namespace nn {

fatal_error::fatal_error(const char *message)
        : std::runtime_error(message) {

        if (logger::log().m_printOnFatal)
                std::cout << message << std::endl;

        logger::log().m_file << logger::pref(LOG_FATAL) << message << std::endl;
}

#ifdef BUILD_CUDA_SUPPORT
cuda_error::cuda_error(const char *message, cudaError_t error)
        : fatal_error(std::format("{} [{}]", message, cudaGetErrorString(error)).data()) {}

cuda_bad_alloc::cuda_bad_alloc(cudaError_t error)
        : cuda_error("Could not allocate memory on the GPU.", error) {}
#endif // BUILD_CUDA_SUPPORT

} // namespace nn
