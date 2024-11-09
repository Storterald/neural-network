#include "Layer.h"

#include "layers/FullyConnectedLayer.h"
#include "Base.h"

std::unique_ptr<ILayer> Layer::create(
        uint32_t previousLayerSize,
        const LayerCreateInfo &layerInfo
) {
        switch (layerInfo.type) {
                case FULLY_CONNECTED:
                        return std::make_unique<FullyConnectedLayer>(previousLayerSize, layerInfo.neuronCount, layerInfo.functionType);
                default:
                        throw Logger::fatal_error("Layer type not recognized.");
        }
}

std::unique_ptr<ILayer> Layer::create(
        uint32_t previousLayerSize,
        const LayerCreateInfo &layerInfo,
        std::ifstream &inputFile
) {
        switch (layerInfo.type) {
                case FULLY_CONNECTED:
                        return std::make_unique<FullyConnectedLayer>(previousLayerSize, layerInfo.neuronCount, layerInfo.functionType, inputFile);
                default:
                        throw Logger::fatal_error("Layer type not recognized.");
        }
}

