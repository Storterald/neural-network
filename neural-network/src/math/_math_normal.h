#pragma once

#include <cstdint>

#include <neural-network/math/math.h>

namespace nn::_math_normal {
        
void sum(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void sum(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void sub(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void sub(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);


void mul(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void mul(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void div(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void div(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void fma(
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]);

void fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]);

void sum(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void sum(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void sub(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void sub(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void mul(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void mul(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void div(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void div(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]);

void fma(
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]);

void fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]);

void tanh(
        uint32_t    size,
        const float data[],
        float       result[]);

void tanh(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]);

void tanh_derivative(
        uint32_t    size,
        const float data[],
        float       result[]);

void tanh_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]);

void ReLU(
        uint32_t    size,
        const float data[],
        float       result[]);

void ReLU(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]);

void ReLU_derivative(
        uint32_t    size,
        const float data[],
        float       result[]);

void ReLU_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]);

void min(
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]);

void min(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]);

void max(
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]);

void max(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]);

void clamp(
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]);

void clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]);

void min(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void min(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void max(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void max(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]);

void clamp(
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]);

void clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]);

void compare(
        uint32_t    size,
        const float first[],
        const float second[],
        bool        *result);

void matvec_r(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]);

void matvec_r(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]);

void matvec_c(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]);

void matvec_c(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]);

void transpose(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]);

void transpose(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]);

void matmul_rc(
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]);

void matmul_rc(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]);

} // namespace nn::_math_normal
