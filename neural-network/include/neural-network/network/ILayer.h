#pragma once

#include <cstdint>

#include <neural-network/types/Vector.h>
#include <neural-network/Base.h>

NN_BEGIN

enum FunctionType : uint32_t {
        TANH,
        RELU

}; // enum FunctionType

enum LayerType : uint32_t {
        FULLY_CONNECTED

}; // enum LayerType

struct LayerCreateInfo {
        LayerType           type;
        FunctionType        functionType;
        uint32_t            neuronCount;

}; // struct LayerCreateInfo

class ILayer {
public:
        [[nodiscard]] virtual Vector forward(const Vector &input) const = 0;
        virtual Vector backward(const Vector &cost, const Vector &input) = 0;

        virtual void encode(std::ostream &file) const = 0;

        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~ILayer() = default;

}; // interface ILayer

NN_END
