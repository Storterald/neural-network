#include "Layer.h"

#include "layers/FullyConnectedLayer.h"

std::unique_ptr<ILayer> Layer::create(
        uint32_t previousLayerSize,
        const LayerCreateInfo &layerInfo
) {
        switch (layerInfo.type) {
                case FULLY_CONNECTED:
                        return std::make_unique<FullyConnectedLayer>(previousLayerSize, layerInfo.neuronCount, layerInfo.functionType);
        }

        // Unreachable, the switch above should cover all possible
        // options present in the FunctionType enum.
        return {};
}

std::unique_ptr<ILayer> Layer::create(
        uint32_t previousLayerSize,
        const LayerCreateInfo &layerInfo,
        std::ifstream &inputFile
) {
        switch (layerInfo.type) {
                case FULLY_CONNECTED:
                        return std::make_unique<FullyConnectedLayer>(previousLayerSize, layerInfo.neuronCount, layerInfo.functionType, inputFile);
        }

        // Unreachable, the switch above should cover all possible
        // options present in the FunctionType enum.
        return {};
}

