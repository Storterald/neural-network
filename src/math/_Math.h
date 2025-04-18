#pragma once

#include <cstdint>

enum MathType {
        MATH_SSE,
        MATH_AVX,
        MATH_AVX512,
        MATH_CUDA
};

template<MathType>
class _Math {
public:
        static void sum(uint32_t size, const float first[], const float second[], float result[]);
        static void sub(uint32_t size, const float first[], const float second[], float result[]);
        static void mul(uint32_t size, const float first[], const float second[], float result[]);
        static void div(uint32_t size, const float first[], const float second[], float result[]);
        static void sum(uint32_t size, const float data[], float scalar, float result[]);
        static void sub(uint32_t size, const float data[], float scalar, float result[]);
        static void mul(uint32_t size, const float data[], float scalar, float result[]);
        static void div(uint32_t size, const float data[], float scalar, float result[]);
        static void tanh(uint32_t size, const float data[], float result[]);
        static void tanh_derivative(uint32_t size, const float data[], float result[]);
        static void ReLU(uint32_t size, const float data[], float result[]);
        static void ReLU_derivative(uint32_t size, const float data[], float result[]);
        static void min(uint32_t size, const float data[], float min, float result[]);
        static void max(uint32_t size, const float data[], float max, float result[]);
        static void clamp(uint32_t size, const float data[], float min, float max, float result[]);
        static void min(uint32_t size, const float first[], const float second[], float result[]);
        static void max(uint32_t size, const float first[], const float second[], float result[]);
        static void clamp(uint32_t size, const float data[], const float min[], const float max[], float result[]);
        static void matvec_mul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[]);

};