#include "_math_normal.h"

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace nn {

void _math_normal::sum(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] + second[i];
}

void _math_normal::sum(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = first[i] + second[i];
}

void _math_normal::sub(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] - second[i];
}

void _math_normal::sub(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = first[i] - second[i];
}

void _math_normal::mul(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] * second[i];
}

void _math_normal::mul(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = first[i] * second[i];
}

void _math_normal::div(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = first[i] / second[i];
}

void _math_normal::div(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = first[i] / second[i];
}

void _math_normal::fma(
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::fmaf(first[i], second[i], third[i]);
}

void _math_normal::fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::fmaf(first[i], second[i], third[i]);
}

void _math_normal::sum(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] + scalar;
}

void _math_normal::sum(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = data[i] + scalar;
}

void _math_normal::sub(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] - scalar;
}

void _math_normal::sub(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = data[i] - scalar;
}

void _math_normal::mul(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] * scalar;
}

void _math_normal::mul(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = data[i] * scalar;
}

void _math_normal::div(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] / scalar;
}

void _math_normal::div(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = data[i] / scalar;
}

void _math_normal::fma(
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::fmaf(first[i], scalar, third[i]);
}

void _math_normal::fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::fmaf(first[i], scalar, third[i]);
}

void _math_normal::tanh(
        uint32_t    size,
        const float data[],
        float       result[]) {

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

void _math_normal::tanh(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i) {
                const float x = data[i];

                if (std::abs(x) >= 4.9f) {
                        result[i] = std::copysign(1.0f, x);
                        continue;
                }

                const float x2 = x * x;
                result[i] = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) /
                            (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }
}

void _math_normal::tanh_derivative(
        uint32_t    size,
        const float data[],
        float       result[]) {

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

                const float tanh = result[i];
                result[i] = 1 - tanh * tanh;
        }
}

void _math_normal::tanh_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        _math_normal::tanh(nn::parallel, size, data, result);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i) {
                const float x = data[i];

                if (x == 0.0f) {
                        result[i] = 1.0f;
                        continue;
                }

                if (std::abs(x) > 4.9f) {
                        result[i] = 0.0f;
                        continue;
                }

                const float tanh = result[i];
                result[i] = 1 - tanh * tanh;
        }
}

void _math_normal::ReLU(
        uint32_t    size,
        const float data[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::max(data[i], 0.0f);
}

void _math_normal::ReLU(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::max(data[i], 0.0f);
}

void _math_normal::ReLU_derivative(
        uint32_t    size,
        const float data[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = data[i] >= 0.0f ? 1.0f : 0.0f;
}

void _math_normal::ReLU_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = data[i] >= 0.0f ? 1.0f : 0.0f;
}

void _math_normal::min(
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::min(data[i], min);
}

void _math_normal::min(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::min(data[i], min);
}

void _math_normal::max(
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::max(data[i], max);
}

void _math_normal::max(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::max(data[i], max);
}

void _math_normal::clamp(
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::clamp(data[i], min, max);
}

void _math_normal::clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::clamp(data[i], min, max);
}

void _math_normal::min(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::min(first[i], second[i]);
}

void _math_normal::min(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::min(first[i], second[i]);
}

void _math_normal::max(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::max(first[i], second[i]);
}

void _math_normal::max(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::max(first[i], second[i]);
}

void _math_normal::clamp(
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]) {

        for (uint32_t i = 0; i < size; ++i)
                result[i] = std::clamp(data[i], min[i], max[i]);
}

void _math_normal::clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i)
                result[i] = std::clamp(data[i], min[i], max[i]);
}

void _math_normal::compare(
        uint32_t    size,
        const float first[],
        const float second[],
        bool        *result) {

        *result = true;
        for (uint32_t i = 0; i < size && *result; ++i)
                *result = first[i] == second[i];
}

void _math_normal::matvec_r(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        for (uint32_t i = 0; i < height; ++i)
                for (uint32_t j = 0; j < width; ++j)
                        result[i] = std::fma(matrix[i * width + j], vector[j], result[i]);
}

void _math_normal::matvec_r(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(height); ++i)
                for (uint32_t j = 0; j < width; ++j)
                        result[i] = std::fma(matrix[i * width + j], vector[j], result[i]);
}

void _math_normal::matvec_c(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        for (uint32_t j = 0; j < width; ++j)
                for (uint32_t i = 0; i < height; ++i)
                        result[i] = std::fma(matrix[j * height + i], vector[j], result[i]);
}

void _math_normal::matvec_c(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t j = 0; j < static_cast<int32_t>(width); ++j)
                for (uint32_t i = 0; i < height; ++i)
                        result[i] = std::fma(matrix[j * height + i], vector[j], result[i]);
}

void _math_normal::transpose(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]) {

        for (uint32_t i = 0; i < height; ++i)
                for (uint32_t j = 0; j < width; ++j)
                        result[j * height + i] = matrix[i * width + j];
}

void _math_normal::transpose(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(height); ++i)
                for (uint32_t j = 0; j < width; ++j)
                        result[j * height + i] = matrix[i * width + j];
}

void _math_normal::matmul_rc(
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]) {

        for (uint32_t i = 0; i < height; ++i)
                for (uint32_t j = 0; j < width2; ++j)
                        for (uint32_t k = 0; k < width; ++k)
                                result[i * width2 + j] = std::fma(rmatrix[i * width + k], cmatrix[j * width + k], result[i * width2 + j]);
}

void _math_normal::matmul_rc(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]) {

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(height); ++i)
                for (uint32_t j = 0; j < width2; ++j)
                        for (uint32_t k = 0; k < width; ++k)
                                result[i * width2 + j] = std::fma(rmatrix[i * width + k], cmatrix[j * width + k], result[i * width2 + j]);
}

} // namespace nn
