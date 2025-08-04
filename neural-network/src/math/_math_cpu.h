#pragma once

#include <cstdint>

#include <neural-network/math/math.h>

#ifdef TARGET_X86_64
#include "_math_simd.h"
#define FUNCTION(__name__, ...) _math_simd:: __name__(__VA_ARGS__)
#else // TARGET_X86_64
#include "_math_normal.h"
#define FUNCTION(__name__, ...) _math_normal:: __name__(__VA_ARGS__)
#endif // TARGET_X86_64

namespace nn::_math_cpu {

inline void sum(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(sum, size, first, second, result);
}

inline void sum(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(sum, nn::parallel, size, first, second, result);
}

inline void sub(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(sub, size, first, second, result);
}

inline void sub(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(sub, nn::parallel, size, first, second, result);
}

inline void mul(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(mul, size, first, second, result);
}

inline void mul(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(mul, nn::parallel, size, first, second, result);
}

inline void div(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(div, size, first, second, result);
}

inline void div(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(div, nn::parallel, size, first, second, result);
}

inline void fma(
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]) {

        FUNCTION(fma, size, first, second, third, result);
}

inline void fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]) {

        FUNCTION(fma, nn::parallel, size, first, second, third, result);
}

inline void sum(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(sum, size, data, scalar, result);
}

inline void sum(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(sum, nn::parallel, size, data, scalar, result);
}

inline void sub(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(sub, size, data, scalar, result);
}

inline void sub(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(sub, nn::parallel, size, data, scalar, result);
}

inline void mul(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(mul, size, data, scalar, result);
}

inline void mul(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(mul, nn::parallel, size, data, scalar, result);
}

inline void div(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(div, size, data, scalar, result);
}

inline void div(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        FUNCTION(div, nn::parallel, size, data, scalar, result);
}

inline void fma(
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]) {

        FUNCTION(fma, size, first, scalar, third, result);
}

inline void fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]) {

        FUNCTION(fma, nn::parallel, size, first, scalar, third, result);
}

inline void tanh(
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(tanh, size, data, result);
}

inline void tanh(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(tanh, nn::parallel, size, data, result);
}

inline void tanh_derivative(
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(tanh_derivative, size, data, result);
}

inline void tanh_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(tanh_derivative, nn::parallel, size, data, result);
}

inline void ReLU(
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(ReLU, size, data, result);
}

inline void ReLU(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(ReLU, nn::parallel, size, data, result);
}

inline void ReLU_derivative(
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(ReLU_derivative, size, data, result);
}

inline void ReLU_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        FUNCTION(ReLU_derivative, nn::parallel, size, data, result);
}

inline void min(
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]) {

        FUNCTION(min, size, data, min, result);
}

inline void min(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]) {

        FUNCTION(min, nn::parallel, size, data, min, result);
}

inline void max(
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]) {

        FUNCTION(max, size, data, max, result);
}

inline void max(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]) {

        FUNCTION(max, nn::parallel, size, data, max, result);
}

inline void clamp(
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]) {

        FUNCTION(clamp, size, data, min, max, result);
}

inline void clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]) {

        FUNCTION(clamp, nn::parallel, size, data, min, max, result);
}

inline void min(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(min, size, first, second, result);
}

inline void min(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(min, nn::parallel, size, first, second, result);
}

inline void max(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(max, size, first, second, result);
}

inline void max(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        FUNCTION(max, nn::parallel, size, first, second, result);
}

inline void clamp(
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]) {

        FUNCTION(clamp, size, data, min, max, result);
}

inline void clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]) {

        FUNCTION(clamp, nn::parallel, size, data, min, max, result);
}

inline void compare(
        uint32_t    size,
        const float first[],
        const float second[],
        bool        *result) {

        FUNCTION(compare, size, first, second, result);
}

inline void compare(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        bool        *result) {

        FUNCTION(compare, size, first, second, result);
}

inline void matvec_r(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        FUNCTION(matvec_r, width, height, matrix, vector, result);
}

inline void matvec_r(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        FUNCTION(matvec_r, nn::parallel, width, height, matrix, vector, result);
}

inline void matvec_c(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        FUNCTION(matvec_c, width, height, matrix, vector, result);
}

inline void matvec_c(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        FUNCTION(matvec_c, nn::parallel, width, height, matrix, vector, result);
}

inline void transpose(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]) {

        FUNCTION(transpose, width, height, matrix, result);
}

inline void transpose(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]) {

        FUNCTION(transpose, nn::parallel, width, height, matrix, result);
}

inline void matmul_rc(
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]) {

        FUNCTION(matmul_rc, width, height, width2, rmatrix, cmatrix, result);
}

inline void matmul_rc(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]) {

        FUNCTION(matmul_rc, nn::parallel, width, height, width2, rmatrix, cmatrix, result);
}

} // namespace nn::_math_cpu
