#pragma once

#ifndef BUILD_CUDA_SUPPORT
#error utils/cuda.h cannot be included without BUILD_CUDA_SUPPORT
#endif // !BUILD_CUDA_SUPPORT

#include <cuda_runtime.h>
#include <driver_types.h>

#include <cstddef> // size_t

#include <neural-network/utils/exceptions.h>

namespace nn::cuda {

// TODO alloc on cpu if out of memory error
inline void *malloc(size_t size)
{
        void *ptr;
        if (const cudaError_t code = cudaMalloc(&ptr, size); code != cudaSuccess)
                throw cuda_bad_alloc(code);
        return ptr;
}

inline void *malloc(size_t size, cudaStream_t stream)
{
        void *ptr;
        if (const cudaError_t code = cudaMallocAsync(&ptr, size, stream); code != cudaSuccess)
                throw cuda_bad_alloc(code);
        return ptr;
}

template<typename T>
inline T *alloc(size_t count = 1)
{
        return (T *)malloc(count * sizeof(T));
}

template<typename T>
inline T *alloc(cudaStream_t stream)
{
        return (T *)malloc(sizeof(T), stream);
}

template<typename T>
inline T *alloc(size_t count, cudaStream_t stream)
{
        return (T *)malloc(count * sizeof(T), stream);
}

inline void free(void *ptr) noexcept
{
        // Error codes won't be validated on buffer destruction. This results
        // in undefined behaviour like a failure on delete.
        cudaFree(ptr);
}

inline void free(void *ptr, cudaStream_t stream) noexcept
{
        cudaFreeAsync(ptr, stream);
}

inline void memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
        if (const cudaError_t code = cudaMemcpy(dst, src, size, kind); code != cudaSuccess)
                throw cuda_error("Could not copy in the GPU.", code);
}

inline void memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream)
{
        if (const cudaError_t code = cudaMemcpyAsync(dst, src, size, kind, stream); code != cudaSuccess)
                throw cuda_error("Could not set memory in the GPU.", code);
}

inline void memset(void *ptr, uint8_t value, size_t size)
{
        if (const cudaError_t code = cudaMemset(ptr, value, size); code != cudaSuccess)
                throw cuda_error("Could not set memory in the GPU.", code);
}

inline void memset(void *ptr, uint8_t value, size_t size, cudaStream_t stream)
{
        if (const cudaError_t code = cudaMemsetAsync(ptr, value, size, stream); code != cudaSuccess)
                throw cuda_error("Could not set memory in the GPU.", code);
}

inline void sync(cudaStream_t stream)
{
        if (const cudaError_t code = cudaStreamSynchronize(stream); code != cudaSuccess)
                throw cuda_error("Could not synchronize the stream with the CPU.", code);
}

inline void sync_all()
{
        if (const cudaError_t code = cudaDeviceSynchronize(); code != cudaSuccess)
                throw cuda_error("Could not synchronize the GPU with the CPU.", code);
}

inline void check_last_error(const char *message)
{
        if (const cudaError_t code = cudaGetLastError(); code != cudaSuccess)
                throw cuda_error(message, code);
}

inline cudaStream_t create_stream()
{
        cudaStream_t stream;
        if (const cudaError_t code = cudaStreamCreate(&stream); code != cudaSuccess)
                throw cuda_error("Could not create CUDA stream.", code);
        return stream;
}

} // namespace nn::cuda