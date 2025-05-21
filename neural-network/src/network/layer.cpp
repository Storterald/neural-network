#include <neural-network/network/layer.h>

#include <iostream> // std::istream
#include <cstdint>
#include <memory>

#include <neural-network/network/layers/fully_connected_layer.h>
#include <neural-network/utils/exceptions.h>
#include <neural-network/base.h>

namespace nn {

std::unique_ptr<layer> layer::create(
        uint32_t                       previousLayerSize,
        const layer_create_info        &layerInfo,
        stream                         stream) {

        switch (layerInfo.type) {
        case FULLY_CONNECTED:
                return std::make_unique<fully_connected_layer>(
                        previousLayerSize, layerInfo.neuronCount,
                        layerInfo.functionType, stream);
        default:
                throw fatal_error("Layer type not recognized.");
        }
}

std::unique_ptr<layer> layer::create(
        uint32_t                       previousLayerSize,
        const layer_create_info        &layerInfo,
        std::istream                   &inputStream,
        stream                         stream) {

        switch (layerInfo.type) {
        case FULLY_CONNECTED:
                return std::make_unique<fully_connected_layer>(
                        previousLayerSize, layerInfo.neuronCount,
                        layerInfo.functionType, inputStream, stream);
        default:
                throw fatal_error("Layer type not recognized.");
        }
}

} // namespace nn
