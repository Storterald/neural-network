#pragma once

#ifndef TARGET_X86_64
#error cannot be included if the architecture is not x86_64
#endif // !TARGET_X86_64

#include <cstdint>

#include <neural-network/utils/simd.h>
#include "_math_normal.h"

namespace nn::_math_simd {

template<typename m = simd::simd>
inline void sum(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                const m sumResult = values + otherValues;
                sumResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sum(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void sub(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                const m subResult = values - otherValues;
                subResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sub(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void mul(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                const m mulResult = values * otherValues;
                mulResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::mul(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void div(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                const m divResult = values / otherValues;
                divResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::div(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void sum(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m scalarValues(scalar);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m sumResult = values + scalarValues;
                sumResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sum(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void sub(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m scalarValues(scalar);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m subResult = values - scalarValues;
                subResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sub(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void mul(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m scalarValues(scalar);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m mulResult = values * scalarValues;
                mulResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::mul(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void div(
        uint32_t           size,
        const float        data[],
        float              scalar,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m scalarValues(scalar);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m divResult = values / scalarValues;
                divResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::div(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void tanh(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m threshold(4.9f);
        const m one(1.0f);
        const m negative(-0.0f);

        const m v28(28.0f);
        const m v378(378.0f);
        const m v3150(3150.0f);
        const m v17325(17325.0f);
        const m v62370(62370.0f);
        const m v135135(135135.0f);

        for (uint32_t i = 0; i < end; i +=  m::width) {
                const m x(&data[i]);
                const m x2 = x * x;

                const m absoluteX           = x.abs();
                const typename m::mask mask = absoluteX >= threshold;
                const m signs               = x & negative;
                const m signedOne           = signs | one;

                m numerator = x2 + v378;
                numerator            = x2.fma(numerator, v17325);
                numerator            = x2.fma(numerator, v135135);
                numerator            = x * numerator;

                m denominator = x2.fma(v28, v3150);
                denominator            = x2.fma(denominator, v62370);
                denominator            = x2.fma(denominator, v135135);

                m tanh = numerator / denominator;
                tanh            = mask(tanh, signedOne);

                tanh.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::tanh(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void tanh_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        _math_simd::tanh(size - end, data, result);

        const m threshold(4.9f);
        const m zero{};
        const m one(1.0f);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m tanhValues(&result[i]);
                m tanhDerivative = tanhValues.fnma(tanhValues, one);

                const typename m::mask neg = values.abs() > threshold;
                tanhDerivative             = neg(tanhDerivative, zero);

                const typename m::mask eq = values == zero;
                tanhDerivative            = eq(tanhDerivative, one);

                tanhDerivative.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::tanh_derivative(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void ReLU(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m zero{};

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m x(&data[i]);
                const m relu = zero.max(x);
                relu.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::ReLU(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void ReLU_derivative(
        uint32_t           size,
        const float        data[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m zero{};
        const m one(1.0f);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m x(&data[i]);
                const typename m::mask pos = x > zero;
                const m reluDerivative     = pos(zero, one);
                reluDerivative.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::ReLU_derivative(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void min(
        uint32_t           size,
        const float        data[],
        float              min,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m minValues(min);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m minResult = values.min(minValues);
                minResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::min(rem, data + end, min, result + end);
}

template<typename m = simd::simd>
inline void max(
        uint32_t           size,
        const float        data[],
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m maxValues(max);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m maxResult = values.max(maxValues);
                maxResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::max(rem, data + end, max, result + end);
}

template<typename m = simd::simd>
inline void clamp(
        uint32_t           size,
        const float        data[],
        float              min,
        float              max,
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m minValues(min);
        const m maxValues(max);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m clamp = minValues.max(maxValues.min(values));
                clamp.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::clamp(rem, data + end, min, max, result + end);
}

template<typename m = simd::simd>
inline void min(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                const m minValues = otherValues.min(values);
                minValues.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::min(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void max(
        uint32_t           size,
        const float        first[],
        const float        second[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                const m maxValues = otherValues.max(values);
                maxValues.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::max(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void clamp(
        uint32_t           size,
        const float        data[],
        const float        min[],
        const float        max[],
        float              result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&data[i]);
                const m minValues(&min[i]);
                const m maxValues(&max[i]);
                const m clamp = minValues.max(maxValues.min(values));
                clamp.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::clamp(rem, data + end, min + end, max + end, result + end);
}

template<typename m = simd::simd>
inline void compare(
        uint32_t           size,
        const float        first[],
        const float        second[],
        bool               *result) {

        const uint32_t end = size & ~(m::width - 1);

        *result = true;

        for (uint32_t i = 0; i < end; i+= m::width) {
                const m values(&first[i]);
                const m otherValues(&second[i]);
                if ((values == otherValues) != m::mask::ones) {
                        *result = false;
                        return;
                }
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::compare(rem, first + end, second + end, result);
}

template<typename m = simd::simd>
inline void matvec_mul(
        uint32_t           width,
        uint32_t           height,
        const float        matrix[],
        const float        vector[],
        float              result[]) {

        const uint32_t end = width & ~(m::width - 1);

        for (uint32_t i = 0; i < height; ++i) {
                m sum{};

                for (uint32_t j = 0; j < end; j+= m::width) {
                        const m values(&matrix[i * width + j]);
                        const m vectorValues(&vector[j]);
                        const m product = values * vectorValues;
                        sum += product;
                }

                result[i] = sum.sum();

                for (uint32_t j = end; j < width; ++j)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}

} // namespace nn::_math_simd
