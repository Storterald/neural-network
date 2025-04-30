#pragma once

#include <cstdint>

#include <neural-network/types/buf.h>

namespace nn::math {

        void sum(
                uint32_t        size,
                const buf       &first,
                const buf       &second,
                buf             &result);

        void sub(
                uint32_t        size,
                const buf       &first,
                const buf       &second,
                buf             &result);

        void mul(
                uint32_t        size,
                const buf       &first,
                const buf       &second,
                buf             &result);

        void div(
                uint32_t        size,
                const buf       &first,
                const buf       &second,
                buf             &result);

        void sum(
                uint32_t         size,
                const buf        &data,
                float            scalar,
                buf              &result);

        void sub(
                uint32_t         size,
                const buf        &data,
                float            scalar,
                buf              &result);

        void mul(
                uint32_t         size,
                const buf        &data,
                float            scalar,
                buf              &result);

        void div(
                uint32_t         size,
                const buf        &data,
                float            scalar,
                buf              &result);

        void tanh(
                uint32_t         size,
                const buf        &data,
                buf              &result);

        void tanh_derivative(
                uint32_t         size,
                const buf        &data,
                buf              &result);

        void ReLU(
                uint32_t         size,
                const buf        &data,
                buf              &result);

        void ReLU_derivative(
                uint32_t         size,
                const buf        &data,
                buf              &result);

        void min(
                uint32_t         size,
                const buf        &data,
                float            min,
                buf              &result);

        void max(
                uint32_t         size,
                const buf        &data,
                float            max,
                buf              &result);

        void clamp(
                uint32_t         size,
                const buf        &data,
                float            min,
                float            max,
                buf              &result);

        void min(
                uint32_t         size,
                const buf        &first,
                const buf        &second,
                buf              &result);

        void max(
                uint32_t         size,
                const buf        &first,
                const buf        &second,
                buf              &result);

        void clamp(
                uint32_t         size,
                const buf        &data,
                const buf        &min,
                const buf        &max,
                buf              &result);

        void compare(
                uint32_t         size,
                const buf        &first,
                const buf        &second,
                bool             *result);

        void matvec_mul(
                uint32_t         width,
                uint32_t         height,
                const buf        &matrix,
                const buf        &vector,
                buf              &result);

} // namespace nn::math
