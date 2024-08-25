#pragma once

#include "Vector.cuh"

namespace Utils {

        void forward(
                float *result,
                const float *input,
                const float *w,
                const float *d,
                uint32_t width,
                uint32_t height
        );

        void backward(
                float *previousCosts,
                float *dw,
                const float *db,
                const float *w,
                const float *input,
                uint32_t width,
                uint32_t height
        );

        void fmaScalar(
                float *result,
                const float *a,
                float scalar,
                const float *c,
                uint32_t size
        );

} // namespace Utils

