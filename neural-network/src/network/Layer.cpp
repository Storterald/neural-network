#include <neural-network/network/Layer.h>

#include <cstdint>
#include <fstream>
#include <memory>

#include <neural-network/network/layers/FullyConnectedLayer.h>
#include <neural-network/network/ILayer.h>
#include <neural-network/utils/Logger.h>

std::unique_ptr<ILayer> Layer::create(
        uint32_t                     previousLayerSize,
        const LayerCreateInfo        &layerInfo) {

        switch (layerInfo.type) {
        case FULLY_CONNECTED:
                return std::make_unique<FullyConnectedLayer>(
                        previousLayerSize, layerInfo.neuronCount,
                        layerInfo.functionType);
        default:
                throw LOGGER_EX("Layer type not recognized.");
        }
}

std::unique_ptr<ILayer> Layer::create(
        uint32_t                     previousLayerSize,
        const LayerCreateInfo        &layerInfo,
        std::ifstream                &inputFile) {

        switch (layerInfo.type) {
        case FULLY_CONNECTED:
                return std::make_unique<FullyConnectedLayer>(
                        previousLayerSize, layerInfo.neuronCount,
                        layerInfo.functionType, inputFile);
        default:
                throw LOGGER_EX("Layer type not recognized.");
        }
}
