#pragma once

#ifdef USE_CUDA
#include "../cuda/Matrix.cuh"
#include "../cuda/FastMath.cuh"
#else
#include "../math/Matrix.h"
#include "../math/FastMath.h"
#endif

#include "../enums/FunctionType.h"
#include "../enums/LayerType.h"

struct LayerCreateInfo {
        LayerType type;
        FunctionType functionType;
        uint32_t neuronCount;

}; // struct LayerCreateInfo

// The interface ILayer, all layers inherit this.
class ILayer {
public:
        [[nodiscard]] virtual Vector forward(const Vector &input) const = 0;
        virtual Vector backward(const Vector &cost, const Vector &input) = 0;

        virtual void encode(std::ofstream &) const = 0;
        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~ILayer() = default;

}; // interface ILayer

inline Vector ApplyActivation(FunctionType functionType, const Vector &input) {
        switch (functionType) {
                case TANH:
                        return Fast::tanh(input);
                case RELU:
                        return Fast::relu(input);
        }

        // Unreachable, the switch above should cover all possible
        // options present in the FunctionType enum.
        return {};
}

inline Vector ApplyActivationDerivative(FunctionType functionType, const Vector &input) {
        switch (functionType) {
                case TANH:
                        return Fast::tanhDerivative(input);
                case RELU:
                        return Fast::reluDerivative(input);
        }

        // Unreachable, the switch above should cover all possible
        // options present in the FunctionType enum.
        return {};
}