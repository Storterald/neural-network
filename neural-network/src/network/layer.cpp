#include <neural-network/network/layer.h>

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

#include <iostream>
#include <cstdint>
#include <memory>

#include <neural-network/network/layers/fully_connected_layer.h>
#include <neural-network/utils/logger.h>

namespace nn {

std::unique_ptr<layer> layer::create(
        uint32_t                       previousLayerSize,
        const layer_create_info        &layerInfo) {

        switch (layerInfo.type) {
        case FULLY_CONNECTED:
                return std::make_unique<fully_connected_layer>(
                        previousLayerSize, layerInfo.neuronCount,
                        layerInfo.functionType);
        default:
                throw LOGGER_EX("Layer type not recognized.");
        }
}

std::unique_ptr<layer> layer::create(
        uint32_t                       previousLayerSize,
        const layer_create_info        &layerInfo,
        std::ifstream                  &inputFile) {

        switch (layerInfo.type) {
        case FULLY_CONNECTED:
                return std::make_unique<fully_connected_layer>(
                        previousLayerSize, layerInfo.neuronCount,
                        layerInfo.functionType, inputFile);
        default:
                throw LOGGER_EX("Layer type not recognized.");
        }
}

#ifdef BUILD_CUDA_SUPPORT
std::unique_ptr<layer> layer::create(
        uint32_t                       previousLayerSize,
        const layer_create_info        &layerInfo,
        cudaStream_t                   stream) {

        switch (layerInfo.type) {
                case FULLY_CONNECTED:
                        return std::make_unique<fully_connected_layer>(
                                previousLayerSize, layerInfo.neuronCount,
                                layerInfo.functionType, stream);
                default:
                        throw LOGGER_EX("Layer type not recognized.");
        }
}

std::unique_ptr<layer> layer::create(
        uint32_t                       previousLayerSize,
        const layer_create_info        &layerInfo,
        std::ifstream                  &inputFile,
        cudaStream_t                   stream) {

        switch (layerInfo.type) {
                case FULLY_CONNECTED:
                        return std::make_unique<fully_connected_layer>(
                                previousLayerSize, layerInfo.neuronCount,
                                layerInfo.functionType, inputFile, stream);
                default:
                        throw LOGGER_EX("Layer type not recognized.");
        }
}
#endif // BUILD_CUDA_SUPPORT

} // namespace nn
