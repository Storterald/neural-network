#pragma once

#ifndef TARGET_X86_64
#error cannot be included if the architecture is not x86_64
#endif // !TARGET_X86_64

#include <cstdint>
#include <cmath>

#include <neural-network/utils/simd.h>
#include "_math_normal.h"

namespace nn::_math_simd {

template<typename m = simd::simd>
inline void sum(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m sum = a + b;
                sum.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sum(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void sum(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m sum = a + b;
                sum.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sum(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void sub(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m sub = a - b;
                sub.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sub(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void sub(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m sub = a - b;
                sub.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sub(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void mul(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m mul = a * b;
                mul.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::mul(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void mul(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m mul = a * b;
                mul.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::mul(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void div(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m div = a / b;
                div.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::div(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void div(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m div = a / b;
                div.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::div(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void fma(
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m c(&third[i]);
                const m fma = a.fma(b, c);
                fma.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::fma(rem, first + end, second + end, third + end, result + end);
}

template<typename m = simd::simd>
inline void fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        const float third[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&first[i]);
                const m b(&second[i]);
                const m c(&third[i]);
                const m fma = a.fma(b, c);
                fma.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::fma(rem, first + end, second + end, third + end, result + end);
}

template<typename m = simd::simd>
inline void sum(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&data[i]);
                const m sum = a + b;
                sum.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sum(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void sum(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&data[i]);
                const m sum = a + b;
                sum.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sum(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void sub(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&data[i]);
                const m sub = a - b;
                sub.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sub(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void sub(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&data[i]);
                const m sub = a - b;
                sub.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::sub(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void mul(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&data[i]);
                const m mul = a * b;
                mul.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::mul(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void mul(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&data[i]);
                const m mul = a * b;
                mul.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::mul(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void div(
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&data[i]);
                const m div = a / b;
                div.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::div(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void div(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       scalar,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&data[i]);
                const m div = a / b;
                div.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::div(rem, data + end, scalar, result + end);
}

template<typename m = simd::simd>
inline void fma(
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m a(&first[i]);
                const m c(&third[i]);
                const m fma = a.fma(b, c);
                fma.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::fma(rem, first + end, scalar, third + end, result + end);
}

template<typename m = simd::simd>
inline void fma(
        parallel_t,
        uint32_t    size,
        const float first[],
        float       scalar,
        const float third[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m b(scalar);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m a(&first[i]);
                const m c(&third[i]);
                const m fma = a.fma(b, c);
                fma.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::fma(rem, first + end, scalar, third + end, result + end);
}

template<typename m = simd::simd>
inline void tanh(
        uint32_t    size,
        const float data[],
        float       result[]) {

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
                numerator   = x2.fma(numerator, v17325);
                numerator   = x2.fma(numerator, v135135);
                numerator   = x * numerator;

                m denominator = x2.fma(v28, v3150);
                denominator   = x2.fma(denominator, v62370);
                denominator   = x2.fma(denominator, v135135);

                m tanh = numerator / denominator;
                tanh   = mask(tanh, signedOne);

                tanh.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::tanh(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void tanh(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

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

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i +=  m::width) {
                const m x(&data[i]);
                const m x2 = x * x;

                const m absoluteX           = x.abs();
                const typename m::mask mask = absoluteX >= threshold;
                const m signs               = x & negative;
                const m signedOne           = signs | one;

                m numerator = x2 + v378;
                numerator   = x2.fma(numerator, v17325);
                numerator   = x2.fma(numerator, v135135);
                numerator   = x * numerator;

                m denominator = x2.fma(v28, v3150);
                denominator   = x2.fma(denominator, v62370);
                denominator   = x2.fma(denominator, v135135);

                m tanh = numerator / denominator;
                tanh   = mask(tanh, signedOne);

                tanh.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::tanh(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void tanh_derivative(
        uint32_t    size,
        const float data[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        _math_simd::tanh<m>(size - end, data, result);

        const m threshold(4.9f);
        const m zero{};
        const m one(1.0f);

        for (uint32_t i = 0; i < end; i += m::width) {
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
inline void tanh_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        _math_simd::tanh<m>(size - end, data, result);

        const m threshold(4.9f);
        const m zero{};
        const m one(1.0f);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
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
        uint32_t    size,
        const float data[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m zero{};

        for (uint32_t i = 0; i < end; i += m::width) {
                const m x(&data[i]);
                const m relu = zero.max(x);
                relu.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::ReLU(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void ReLU(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m zero{};

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m x(&data[i]);
                const m relu = zero.max(x);
                relu.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::ReLU(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void ReLU_derivative(
        uint32_t    size,
        const float data[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m zero{};
        const m one(1.0f);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m x(&data[i]);
                const typename m::mask pos = x > zero;
                const m reluDerivative     = pos(zero, one);
                reluDerivative.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::ReLU_derivative(rem, data + end, result + end);
}

template<typename m = simd::simd>
inline void ReLU_derivative(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m zero{};
        const m one(1.0f);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
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
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m minValues(min);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m values(&data[i]);
                const m minResult = values.min(minValues);
                minResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::min(rem, data + end, min, result + end);
}

template<typename m = simd::simd>
inline void min(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m minValues(min);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m values(&data[i]);
                const m minResult = values.min(minValues);
                minResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::min(rem, data + end, min, result + end);
}


template<typename m = simd::simd>
inline void max(
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m maxValues(max);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m values(&data[i]);
                const m maxResult = values.max(maxValues);
                maxResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::max(rem, data + end, max, result + end);
}

template<typename m = simd::simd>
inline void max(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       max,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m maxValues(max);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m values(&data[i]);
                const m maxResult = values.max(maxValues);
                maxResult.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::max(rem, data + end, max, result + end);
}

template<typename m = simd::simd>
inline void clamp(
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m minValues(min);
        const m maxValues(max);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m values(&data[i]);
                const m clamp = minValues.max(maxValues.min(values));
                clamp.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::clamp(rem, data + end, min, max, result + end);
}

template<typename m = simd::simd>
inline void clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        float       min,
        float       max,
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        const m minValues(min);
        const m maxValues(max);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m values(&data[i]);
                const m clamp = minValues.max(maxValues.min(values));
                clamp.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::clamp(rem, data + end, min, max, result + end);
}

template<typename m = simd::simd>
inline void min(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m values(&first[i]);
                const m other(&second[i]);
                const m minValues = other.min(values);
                minValues.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::min(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void min(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m values(&first[i]);
                const m other(&second[i]);
                const m minValues = other.min(values);
                minValues.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::min(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void max(
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
                const m values(&first[i]);
                const m other(&second[i]);
                const m maxValues = other.max(values);
                maxValues.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::max(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void max(
        parallel_t,
        uint32_t    size,
        const float first[],
        const float second[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
                const m values(&first[i]);
                const m other(&second[i]);
                const m maxValues = other.max(values);
                maxValues.store(&result[i]);
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::max(rem, first + end, second + end, result + end);
}

template<typename m = simd::simd>
inline void clamp(
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

        for (uint32_t i = 0; i < end; i += m::width) {
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
inline void clamp(
        parallel_t,
        uint32_t    size,
        const float data[],
        const float min[],
        const float max[],
        float       result[]) {

        const uint32_t end = size & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(end); i += m::width) {
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
        uint32_t    size,
        const float first[],
        const float second[],
        bool        *result) {

        const uint32_t end = size & ~(m::width - 1);

        *result = true;

        for (uint32_t i = 0; i < end; i += m::width) {
                const m values(&first[i]);
                const m other(&second[i]);
                if ((values == other) != m::mask::ones) {
                        *result = false;
                        return;
                }
        }

        if (const uint32_t rem = size - end; rem != 0)
                _math_normal::compare(rem, first + end, second + end, result);
}

template<typename m = simd::simd>
inline void matvec_r(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        const uint32_t end = width & ~(m::width - 1);

        for (uint32_t i = 0; i < height; ++i) {
                m sum{};

                for (uint32_t j = 0; j < end; j+= m::width) {
                        const m values(&matrix[i * width + j]);
                        const m other(&vector[j]);
                        sum = values.fma(other, sum);
                }

                result[i] = sum.sum();

                for (uint32_t j = end; j < width; ++j)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}

template<typename m = simd::simd>
inline void matvec_r(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        const uint32_t end = width & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(height); ++i) {
                m sum{};

                for (uint32_t j = 0; j < end; j+= m::width) {
                        const m values(&matrix[i * width + j]);
                        const m other(&vector[j]);
                        sum = values.fma(other, sum);
                }

                result[i] = sum.sum();

                for (uint32_t j = end; j < width; ++j)
                        result[i] += matrix[i * width + j] * vector[j];
        }
}

template<typename m = simd::simd>
inline void matvec_c(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        const uint32_t end = height & ~(m::width - 1);

        for (uint32_t i = 0; i < width; ++i) {
                const m scalar(vector[i]);

                for (uint32_t j = 0; j < end; j += m::width) {
                        const m values(&matrix[i * height + j]);
                        const m res(&result[j]);
                        values.fma(scalar, res).store(&result[j]);
                }

                for (uint32_t j = end; j < height; ++j)
                        result[j] = std::fma(matrix[i * height + j], vector[i], result[j]);
        }
}

template<typename m = simd::simd>
inline void matvec_c(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        const float vector[],
        float       result[]) {

        const uint32_t end = height & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(width); ++i) {
                const m scalar(vector[i]);

                for (uint32_t j = 0; j < end; j += m::width) {
                        const m values(&matrix[i * height + j]);
                        const m res(&result[j]);
                        values.fma(scalar, res).store(&result[j]);
                }

                for (uint32_t j = end; j < height; ++j)
                        result[j] = std::fma(matrix[i * height + j], vector[i], result[j]);
        }
}

inline void transpose(
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]) {

        // TODO
        _math_normal::transpose(width, height, matrix, result);
}

inline void transpose(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        const float matrix[],
        float       result[]) {

        // TODO
        _math_normal::transpose(nn::parallel, width, height, matrix, result);
}

template<typename m = simd::simd>
inline void matmul_rc(
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]) {

        const uint32_t height2 = width;
        const uint32_t end     = width & ~(m::width - 1);

        for (uint32_t i = 0; i < height; ++i) {
                for (uint32_t j = 0; j < width2; ++j) {
                        const uint32_t idx = i * width2 + j;
                        m sum{};

                        for (uint32_t k = 0; k < end; k += m::width) {
                                const m values(&rmatrix[i * width + k]);
                                const m other(&cmatrix[j * height2 + k]);
                                sum = values.fma(other, sum);
                        }

                        result[idx] = sum.sum();

                        for (uint32_t k = end; k < width; ++k)
                                result[idx] = std::fma(rmatrix[i * width + k], cmatrix[j * height2 + k], result[idx]);
                }
        }
}

template<typename m = simd::simd>
inline void matmul_rc(
        parallel_t,
        uint32_t    width,
        uint32_t    height,
        uint32_t    width2,
        const float rmatrix[],
        const float cmatrix[],
        float       result[]) {

        const uint32_t height2 = width;
        const uint32_t end     = width & ~(m::width - 1);

#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(height); ++i) {
                for (uint32_t j = 0; j < width2; ++j) {
                        const uint32_t idx = i * width2 + j;
                        m sum{};

                        for (uint32_t k = 0; k < end; k += m::width) {
                                const m values(&rmatrix[i * width + k]);
                                const m other(&cmatrix[j * height2 + k]);
                                sum = values.fma(other, sum);
                        }

                        result[idx] = sum.sum();

                        for (uint32_t k = end; k < width; ++k)
                                result[idx] = std::fma(rmatrix[i * width + k], cmatrix[j * height2 + k], result[idx]);
                }
        }
}

} // namespace nn::_math_simd
