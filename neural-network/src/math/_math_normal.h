#pragma once

#include <cstdint>

namespace nn::_math_normal {
        
        void sum(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]);

        void sub(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]);

        void mul(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]);

        void div(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]);

        void sum(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]);

        void sub(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]);

        void mul(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]);

        void div(
                uint32_t           size,
                const float        data[],
                float              scalar,
                float              result[]);

        void tanh(
                uint32_t           size,
                const float        data[],
                float              result[]);

        void tanh_derivative(
                uint32_t           size,
                const float        data[],
                float              result[]);

        void ReLU(
                uint32_t           size,
                const float        data[],
                float              result[]);

        void ReLU_derivative(
                uint32_t           size,
                const float        data[],
                float              result[]);

        void min(
                uint32_t           size,
                const float        data[],
                float              min,
                float              result[]);

        void max(
                uint32_t           size,
                const float        data[],
                float              max,
                float              result[]);

        void clamp(
                uint32_t           size,
                const float        data[],
                float              min,
                float              max,
                float              result[]);

        void min(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]);

        void max(
                uint32_t           size,
                const float        first[],
                const float        second[],
                float              result[]);

        void clamp(
                uint32_t           size,
                const float        data[],
                const float        min[],
                const float        max[],
                float              result[]);

        void compare(
                uint32_t           size,
                const float        first[],
                const float        second[],
                bool               *result);

        void matvec_mul(
                uint32_t           width,
                uint32_t           height,
                const float        matrix[],
                const float        vector[],
                float              result[]);

} // namespace nn::_math_normal
