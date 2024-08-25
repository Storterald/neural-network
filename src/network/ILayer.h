#pragma once

#ifdef USE_CUDA
#include "../cuda/Vector.cuh"
#else
#include "../math/Vector.h"
#endif

#include "../enums/FunctionType.h"
#include "../enums/LayerType.h"

struct LayerCreateInfo {
        LayerType type;
        FunctionType functionType;
        uint32_t neuronCount;
};

// The interface ILayer, all layers inherit this.
class ILayer {
public:
        [[nodiscard]] virtual Vector forward(const Vector &input) const = 0;
        virtual Vector backward(const Vector &cost, const Vector &input) = 0;

        virtual void encode(std::ofstream &) const = 0;
        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~ILayer() = default;

}; // interface ILayer

// Generic implementation of the Layer class
template<LayerType, FunctionType> class Layer : public ILayer {
        Layer(uint32_t previousLayerSize, uint32_t layerSize);
        explicit Layer(std::ifstream &encodedData);
};