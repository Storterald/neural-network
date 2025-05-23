#include "_math_normal.h"

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace nn {

void _math_normal::sum(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] + second[i];
}

void _math_normal::sub(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] - second[i];
}

void _math_normal::mul(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] * second[i];
}

void _math_normal::div(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] / second[i];
}

void _math_normal::sum(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] + scalar;
}

void _math_normal::sub(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] - scalar;
}

void _math_normal::mul(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] * scalar;
}

void _math_normal::div(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] / scalar;
}

void _math_normal::tanh(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        // Around 6-8 times faster than std::tanh in release mode
        for (uint32_t i = 0; i < size; ++i) {
                const float x = data[i];

                // For |x| > 4.9, tanh(x) is approximately 1 or -1. Returns at 4.9
                // since it's the highest decimal the output isn't higher than 1
                if (std::abs(x) >= 4.9f) {
                        result[i] = std::copysign(1.0f, x);
                        continue;
                }

                const float x2 = x * x;

                // 7th-order approximation (accurate within 0.000085).
                // https://www.desmos.com/calculator/2myik1oe4x
                result[i] = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) /
                            (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }
}

void _math_normal::tanh_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        // Uses the _math<>::tanh to calculate the derivative way faster than
        // with using std::tanh, also more accurate than tanhFast.
        _math_normal::tanh(size, data, result);
        for (uint32_t i = 0; i < size; ++i) {
                const float x = data[i];

                // tanh'(0) = 1
                if (x == 0.0f) {
                        result[i] = 1.0f;
                        continue;
                }

                // After this point, the approximation of the derivative goes negative.
                // Instead, the tanh' goes closer and closer to 0 (accurate within
                // 0.000170). Returns at 4.9 since it's the highest decimal the
                // value is positive.
                if (std::abs(x) > 4.9f) {
                        result[i] = 0.0f;
                        continue;
                }

                // Tanh is stored in the result[].
                const float tanh = result[i];
                result[i] = 1 - tanh * tanh;
        }
}

void _math_normal::ReLU(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::max(data[i], 0.0f);
}

void _math_normal::ReLU_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] >= 0.0f ? 1.0f : 0.0f;
}

void _math_normal::min(
        uint32_t           size,
        const float        data[],
        float              min,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::min(data[i], min);
}

void _math_normal::max(
        uint32_t           size,
        const float        data[],
        float              max,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::max(data[i], max);
}

void _math_normal::clamp(
        uint32_t           size,
        const float        data[],
        float              min,
        float              max,
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::clamp(data[i], min, max);
}

void _math_normal::min(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::min(first[i], second[i]);
}

void _math_normal::max(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::max(first[i], second[i]);
}

void _math_normal::clamp(
        uint32_t           size,
        const float        data[],
        const float        min[],
        const float        max[],
        float              result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::clamp(data[i], min[i], max[i]);
}

void _math_normal::compare(
        uint32_t           size,
        const float        first[],
        const float        second[],
        bool               *result) {

        *result = true;
        for (uint32_t i = 0; i < size && *result; ++i)
                *result = first[i] == second[i];
}

void _math_normal::matvec_mul(
        uint32_t           width,
        uint32_t           height,
        const float        matrix[],
        const float        vector[],
        float              result[]) {

        // Matrix product visualizer:
        // http://matrixmultiplication.xyz
        for (uint32_t i = 0; i < height; ++i)
                for (uint32_t j = 0; j < width; ++j)
                        result[i] += matrix[i * width + j] * vector[j];
}

} // namespace nn
