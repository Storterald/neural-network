#pragma once

#include <iostream>
#include <cstdint>
#include <memory>

#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

enum function_type : uint32_t {
        TANH,
        RELU

}; // enum function_type

enum layer_type : uint32_t {
        FULLY_CONNECTED

}; // enum layer_type

struct layer_create_info {
        layer_type           type;
        function_type        functionType;
        uint32_t             neuronCount;

}; // struct layer_create_info

class layer {
public:
        static std::unique_ptr<layer> create(
                uint32_t                       previousLayerSize,
                const layer_create_info        &layerInfo,
                stream                         stream = invalid_stream);

        static std::unique_ptr<layer> create(
                uint32_t                       previousLayerSize,
                const layer_create_info        &layerInfo,
                std::istream                   &inputStream,
                stream                         stream = invalid_stream);

        [[nodiscard]] virtual vector forward(const vector &input) const = 0;
        virtual vector backward(const vector &cost, const vector &input) = 0;

        virtual void encode(std::ostream &file) const = 0;

        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~layer() = default;

}; // interface layer

} // namespace nn
