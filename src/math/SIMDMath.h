#pragma once

#include <cstdint>

namespace SIMDMath {

        namespace AVX512 {
                void sum(uint32_t size, const float first[], const float second[], float result[]);
                void sub(uint32_t size, const float first[], const float second[], float result[]);
                void mul(uint32_t size, const float first[], const float second[], float result[]);
                void div(uint32_t size, const float first[], const float second[], float result[]);
                void sum(uint32_t size, const float first[], float scalar, float result[]);
                void sub(uint32_t size, const float first[], float scalar, float result[]);
                void mul(uint32_t size, const float first[], float scalar, float result[]);
                void div(uint32_t size, const float first[], float scalar, float result[]);
                void tanh(uint32_t size, const float data[], float result[]);
                void tanhDerivative(uint32_t size, const float data[], float result[]);
                void ReLU(uint32_t size, const float data[], float result[]);
                void ReLUDerivative(uint32_t size, const float data[], float result[]);
                void min(uint32_t size, const float data[], float min, float result[]);
                void max(uint32_t size, const float data[], float max, float result[]);
                void clamp(uint32_t size, const float data[], float min, float max, float result[]);
                void min(uint32_t size, const float first[], const float second[], float result[]);
                void max(uint32_t size, const float first[], const float second[], float result[]);
                void clamp(uint32_t size, const float data[], const float min[], const float max[], float result[]);
                void matrixVectorMul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[]);
        }

        namespace AVX {
                void sum(uint32_t size, const float first[], const float second[], float result[]);
                void sub(uint32_t size, const float first[], const float second[], float result[]);
                void mul(uint32_t size, const float first[], const float second[], float result[]);
                void div(uint32_t size, const float first[], const float second[], float result[]);
                void sum(uint32_t size, const float first[], float scalar, float result[]);
                void sub(uint32_t size, const float first[], float scalar, float result[]);
                void mul(uint32_t size, const float first[], float scalar, float result[]);
                void div(uint32_t size, const float first[], float scalar, float result[]);
                void tanh(uint32_t size, const float data[], float result[]);
                void tanhDerivative(uint32_t size, const float data[], float result[]);
                void ReLU(uint32_t size, const float data[], float result[]);
                void ReLUDerivative(uint32_t size, const float data[], float result[]);
                void min(uint32_t size, const float data[], float min, float result[]);
                void max(uint32_t size, const float data[], float max, float result[]);
                void clamp(uint32_t size, const float data[], float min, float max, float result[]);
                void min(uint32_t size, const float first[], const float second[], float result[]);
                void max(uint32_t size, const float first[], const float second[], float result[]);
                void clamp(uint32_t size, const float data[], const float min[], const float max[], float result[]);
                void matrixVectorMul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[]);
        }

        namespace SSE {
                void sum(uint32_t size, const float first[], const float second[], float result[]);
                void sub(uint32_t size, const float first[], const float second[], float result[]);
                void mul(uint32_t size, const float first[], const float second[], float result[]);
                void div(uint32_t size, const float first[], const float second[], float result[]);
                void sum(uint32_t size, const float first[], float scalar, float result[]);
                void sub(uint32_t size, const float first[], float scalar, float result[]);
                void mul(uint32_t size, const float first[], float scalar, float result[]);
                void div(uint32_t size, const float first[], float scalar, float result[]);
                void tanh(uint32_t size, const float data[], float result[]);
                void tanhDerivative(uint32_t size, const float data[], float result[]);
                void ReLU(uint32_t size, const float data[], float result[]);
                void ReLUDerivative(uint32_t size, const float data[], float result[]);
                void min(uint32_t size, const float data[], float min, float result[]);
                void max(uint32_t size, const float data[], float max, float result[]);
                void clamp(uint32_t size, const float data[], float min, float max, float result[]);
                void min(uint32_t size, const float first[], const float second[], float result[]);
                void max(uint32_t size, const float first[], const float second[], float result[]);
                void clamp(uint32_t size, const float data[], const float min[], const float max[], float result[]);
                void matrixVectorMul(uint32_t width, uint32_t height, const float matrix[], const float vector[], float result[]);
        }

}