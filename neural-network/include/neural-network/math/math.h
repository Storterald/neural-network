#pragma once

#include <cstdint>

#include <neural-network/types/buf.h>

namespace nn {

struct parallel_t {};
static constexpr parallel_t parallel{};

} // namespace nn

namespace nn::math {

template<typename T>
void sum(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        buf<T>       &result);

template<typename T>
void sub(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        buf<T>       &result);

template<typename T>
void mul(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        buf<T>       &result);

template<typename T>
void div(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        buf<T>       &result);

template<typename T>
void fma(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        const buf<T> &third,
        buf<T>       &result);

template<typename T>
void sum(
        uint32_t     size,
        const buf<T> &data,
        float        scalar,
        buf<T>       &result);

template<typename T>
void sub(
        uint32_t     size,
        const buf<T> &data,
        float        scalar,
        buf<T>       &result);

template<typename T>
void mul(
        uint32_t     size,
        const buf<T> &data,
        float        scalar,
        buf<T>       &result);

template<typename T>
void div(
        uint32_t     size,
        const buf<T> &data,
        float        scalar,
        buf<T>       &result);

template<typename T>
void fma(
        uint32_t     size,
        const buf<T> &first,
        float        scalar,
        const buf<T> &third,
        buf<T>       &result);

template<typename T>
void tanh(
        uint32_t     size,
        const buf<T> &data,
        buf<T>       &result);

template<typename T>
void tanh_derivative(
        uint32_t     size,
        const buf<T> &data,
        buf<T>       &result);

template<typename T>
void ReLU(
        uint32_t     size,
        const buf<T> &data,
        buf<T>       &result);

template<typename T>
void ReLU_derivative(
        uint32_t     size,
        const buf<T> &data,
        buf<T>       &result);

template<typename T>
void min(
        uint32_t     size,
        const buf<T> &data,
        float        min,
        buf<T>       &result);

template<typename T>
void max(
        uint32_t     size,
        const buf<T> &data,
        float        max,
        buf<T>       &result);

template<typename T>
void clamp(
        uint32_t     size,
        const buf<T> &data,
        float        min,
        float        max,
        buf<T>       &result);

template<typename T>
void min(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        buf<T>       &result);

template<typename T>
void max(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        buf<T>       &result);

template<typename T>
void clamp(
        uint32_t     size,
        const buf<T> &data,
        const buf<T> &min,
        const buf<T> &max,
        buf<T>       &result);

template<typename T>
void compare(
        uint32_t     size,
        const buf<T> &first,
        const buf<T> &second,
        bool         *result);

template<typename T>
void matvec_r(
        uint32_t     width,
        uint32_t     height,
        const buf<T> &matrix,
        const buf<T> &vec,
        buf<T>       &result);

template<typename T>
void matvec_c(
        uint32_t     width,
        uint32_t     height,
        const buf<T> &matrix,
        const buf<T> &vec,
        buf<T>       &result);

template<typename T>
void transpose(
        uint32_t     width,
        uint32_t     height,
        const buf<T> &matrix,
        buf<T>       &result);

template<typename T>
void matmul_rc(
        uint32_t     width,
        uint32_t     height,
        uint32_t     width2,
        const buf<T> &rmatrix,
        const buf<T> &cmatrix,
        buf<T>       &result);

} // namespace nn::math
