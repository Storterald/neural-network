#pragma once

#ifndef BUILD_CUDA_SUPPORT
#error math/_math_cuda.h cannot be included without BUILD_CUDA_SUPPORT
#endif // !BUILD_CUDA_SUPPORT

#include <cstdint>

#include <neural-network/base.h>

namespace nn::_math_cuda {
        
void sum(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[],
        stream_t    stream);

void sub(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[],
        stream_t    stream);

void mul(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[],
        stream_t    stream);

void div(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[],
        stream_t    stream);

void fma(
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[],
        stream_t    stream);

void sum(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[],
        stream_t    stream);

void sub(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[],
        stream_t    stream);

void mul(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[],
        stream_t    stream);

void div(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[],
        stream_t    stream);

void fma(
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[],
        stream_t    stream);

void tanh(
        uint32_t    size,
        const float data[],
        float       result[],
        stream_t    stream);

void tanh_derivative(
        uint32_t    size,
        const float data[],
        float       result[],
        stream_t    stream);

void ReLU(
        uint32_t    size,
        const float data[],
        float       result[],
        stream_t    stream);

void ReLU_derivative(
        uint32_t    size,
        const float data[],
        float       result[],
        stream_t    stream);

void min(
        uint32_t    size,
        const float data[],
        float       min,
        float       result[],
        stream_t    stream);

void max(
        uint32_t    size,
        const float data[],
        float       max,
        float       result[],
        stream_t    stream);

void clamp(
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[],
        stream_t    stream);

void min(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[],
        stream_t    stream);

void max(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[],
        stream_t    stream);

void clamp(
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[],
        stream_t    stream);

void compare(
        uint32_t    size,
        const float first[],
        const float second[],
        bool        *result,
        stream_t    stream);

void matvec_r(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[],
        stream_t    stream);

void matvec_c(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[],
        stream_t    stream);

void transpose(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[],
        stream_t    stream);

void matmul_rc(
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[],
        stream_t    stream);

} // namespace nn::_math_cuda
