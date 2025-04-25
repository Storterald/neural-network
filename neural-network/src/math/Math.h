#pragma once

#include "../types/Data.h"

namespace Math {
        void sum(
                uint32_t         size,
                const Data       &first,
                const Data       &second,
                Data             &result);

        void sub(
                uint32_t         size,
                const Data       &first,
                const Data       &second,
                Data             &result);

        void mul(
                uint32_t         size,
                const Data       &first,
                const Data       &second,
                Data             &result);

        void div(
                uint32_t         size,
                const Data       &first,
                const Data       &second,
                Data             &result);

        void sum(
                uint32_t          size,
                const Data        &data,
                float             scalar,
                Data              &result);

        void sub(
                uint32_t          size,
                const Data        &data,
                float             scalar,
                Data              &result);

        void mul(
                uint32_t          size,
                const Data        &data,
                float             scalar,
                Data              &result);

        void div(
                uint32_t          size,
                const Data        &data,
                float             scalar,
                Data              &result);

        void tanh(
                uint32_t          size,
                const Data        &data,
                Data              &result);

        void tanh_derivative(
                uint32_t          size,
                const Data        &data,
                Data              &result);

        void ReLU(
                uint32_t          size,
                const Data        &data,
                Data              &result);

        void ReLU_derivative(
                uint32_t          size,
                const Data        &data,
                Data              &result);

        void min(
                uint32_t          size,
                const Data        &data,
                float             min,
                Data              &result);

        void max(
                uint32_t          size,
                const Data        &data,
                float             max,
                Data              &result);

        void clamp(
                uint32_t          size,
                const Data        &data,
                float             min,
                float             max,
                Data              &result);

        void min(
                uint32_t          size,
                const Data        &first,
                const Data        &second,
                Data              &result);

        void max(
                uint32_t          size,
                const Data        &first,
                const Data        &second,
                Data              &result);

        void clamp(
                uint32_t          size,
                const Data        &data,
                const Data        &min,
                const Data        &max,
                Data              &result);

        void matvec_mul(
                uint32_t          width,
                uint32_t          height,
                const Data        &matrix,
                const Data        &vector,
                Data              &result);

} // namespace Math
