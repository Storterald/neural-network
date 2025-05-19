#pragma once

#include <cstdint>

#include <neural-network/base.h>

namespace nn {

class _math_cuda {
public:
        static void sum(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        static void sub(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        static void mul(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        static void div(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        static void sum(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        static void sub(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        static void mul(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        static void div(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        static void tanh(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        static void tanh_derivative(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        static void ReLU(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        static void ReLU_derivative(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        static void min(
                uint32_t           size,
                const float        data[],
                float              min,
                float              result[],
                stream             stream);

        static void max(
                uint32_t           size,
                const float        data[],
                float              max,
                float              result[],
                stream             stream);

        static void clamp(
                uint32_t           size,
                const float        data[],
                float              min,
                float              max,
                float              result[],
                stream             stream);

        static void min(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        static void max(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        static void clamp(
                uint32_t           size,
                const float        data[],
                const float        min[],
                const float        max[],
                float              result[],
                stream             stream);

        static void compare(
                uint32_t           size,
                const float        first[],
                const float        second[],
                bool               *result,
                stream             stream);

        static void matvec_mul(
                uint32_t           width,
                uint32_t           height,
                const float        matrix[],
                const float        vector[],
                float              result[],
                stream             stream);
};

} // namespace nn