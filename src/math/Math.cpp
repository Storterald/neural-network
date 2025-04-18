#include "Math.h"

#include "../intrinsic/Intrinsic.h"
#include "_Math.h"
#include "Base.h"

void Math::sum(uint32_t size, const float first[], const float second[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::sum(size, first, second, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::sum(size, first, second, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::sum(size, first, second, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::sum(size, first, second, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = first[i] + second[i];
        }
}

void Math::sub(uint32_t size, const float first[], const float second[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::sub(size, first, second, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::sub(size, first, second, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::sub(size, first, second, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::sub(size, first, second, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = first[i] - second[i];
        }
}

void Math::mul(uint32_t size, const float first[], const float second[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::mul(size, first, second, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::mul(size, first, second, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::mul(size, first, second, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::mul(size, first, second, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = first[i] * second[i];
        }
}

void Math::div(uint32_t size, const float first[], const float second[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::div(size, first, second, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::div(size, first, second, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::div(size, first, second, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::div(size, first, second, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = first[i] / second[i];
        }
}

void Math::sum(uint32_t size, const float data[], float scalar, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::sum(size, data, scalar, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::sum(size, data, scalar, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::sum(size, data, scalar, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::sum(size, data, scalar, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = data[i] + scalar;
        }
}

void Math::sub(uint32_t size, const float data[], float scalar, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::sub(size, data, scalar, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::sub(size, data, scalar, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::sub(size, data, scalar, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::sub(size, data, scalar, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = data[i] - scalar;
        }
}

void Math::mul(uint32_t size, const float data[], float scalar, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::mul(size, data, scalar, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::mul(size, data, scalar, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::mul(size, data, scalar, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::mul(size, data, scalar, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = data[i] * scalar;
        }
}

void Math::div(uint32_t size, const float data[], float scalar, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::div(size, data, scalar, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::div(size, data, scalar, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::div(size, data, scalar, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::div(size, data, scalar, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = data[i] / scalar;
        }
}

void Math::tanh(uint32_t size, const float data[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::tanh(size, data, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::tanh(size, data, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::tanh(size, data, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::tanh(size, data, result);
        case SIMD_UNSUPPORTED:
                // Done later to avoid high indentation.
                break;
        }

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

void Math::tanh_derivative(uint32_t size, const float data[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::tanh_derivative(size, data, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::tanh_derivative(size, data, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::tanh_derivative(size, data, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::tanh_derivative(size, data, result);
        case SIMD_UNSUPPORTED:
                // Done later to avoid high indentation.
                break;
        }

        // Uses the tanhFast to calculate the derivative way faster than
        // with using std::tanh, also more accurate than tanhFast.
        Math::tanh(size, data, result);
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

void Math::ReLU(uint32_t size, const float data[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::ReLU(size, data, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::ReLU(size, data, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::ReLU(size, data, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::ReLU(size, data, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::min(data[i], 0.0f);
        }
}

void Math::ReLU_derivative(uint32_t size, const float data[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::ReLU_derivative(size, data, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::ReLU_derivative(size, data, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::ReLU_derivative(size, data, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::ReLU_derivative(size, data, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = data[i] >= 0.0f ? 1.0f : 0.0f;
        }
}

void Math::min(uint32_t size, const float data[], float min, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::min(size, data, min, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::min(size, data, min, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::min(size, data, min, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::min(size, data, min, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::min(data[i], min);
        }
}

void Math::max(uint32_t size, const float data[], float max, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::max(size, data, max, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::max(size, data, max, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::max(size, data, max, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::max(size, data, max, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::max(data[i], max);
        }
}

void Math::clamp(uint32_t size, const float data[], float min, float max, float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::clamp(size, data, min, max, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::clamp(size, data, min, max, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::clamp(size, data, min, max, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::clamp(size, data, min, max, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::clamp(data[i], min, max);
        }
}

void Math::min(uint32_t size, const float first[], const float second[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::min(size, first, second, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::min(size, first, second, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::min(size, first, second, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::min(size, first, second, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::min(first[i], second[i]);
        }
}

void Math::max(uint32_t size, const float first[], const float second[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::max(size, first, second, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::max(size, first, second, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::max(size, first, second, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::max(size, first, second, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::max(first[i], second[i]);
        }
}

void Math::clamp(uint32_t size, const float data[], const float min[], const float max[], float result[])
{
        if (size >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::clamp(size, data, min, max, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::clamp(size, data, min, max, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::clamp(size, data, min, max, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::clamp(size, data, min, max, result);
        case SIMD_UNSUPPORTED:
                for (uint32_t i = 0; i < size; ++i)
                        result[i] = std::clamp(data[i], min[i], max[i]);
        }
}

void Math::matvec_mul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[])
{
        if (width * height >= CUDA_MINIMUM)
                return _Math<MATH_CUDA>::matvec_mul(width, height, matrix, vector, result);

        switch (Intrinsic::support()) {
        case SIMD_AVX512:
                return _Math<MATH_AVX512>::matvec_mul(width, height, matrix, vector, result);
        case SIMD_AVX:
                return _Math<MATH_AVX>::matvec_mul(width, height, matrix, vector, result);
        case SIMD_SSE:
                return _Math<MATH_SSE>::matvec_mul(width, height, matrix, vector, result);
        case SIMD_UNSUPPORTED:
                // Matrix product visualizer:
                // http://matrixmultiplication.xyz
                for (uint32_t i = 0; i < height; ++i)
                        for (uint32_t j = 0; j < width; ++j)
                                result[i] += matrix[i * width + j] * vector[j];
        }
}