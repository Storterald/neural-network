#pragma once

#include <cstdint>

#include <neural-network/base.h>

namespace nn::_math_cuda {
        
        void sum(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        void sub(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        void mul(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        void div(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        void sum(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        void sub(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        void mul(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        void div(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[],
                stream             stream);

        void tanh(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        void tanh_derivative(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        void ReLU(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        void ReLU_derivative(
                uint32_t           size,
                const float        data[],
                float              result[],
                stream             stream);

        void min(
                uint32_t           size,
                const float        data[],
                float              min,
                float              result[],
                stream             stream);

        void max(
                uint32_t           size,
                const float        data[],
                float              max,
                float              result[],
                stream             stream);

        void clamp(
                uint32_t           size,
                const float        data[],
                float              min,
                float              max,
                float              result[],
                stream             stream);

        void min(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        void max(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[],
                stream             stream);

        void clamp(
                uint32_t           size,
                const float        data[],
                const float        min[],
                const float        max[],
                float              result[],
                stream             stream);

        void compare(
                uint32_t           size,
                const float        first[],
                const float        second[],
                bool               *result,
                stream             stream);

        void matvec_mul(
                uint32_t           width,
                uint32_t           height,
                const float        matrix[],
                const float        vector[],
                float              result[],
                stream             stream);

} // namespace nn::_math_cuda
