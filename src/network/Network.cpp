#include "Network.h"

#include <fstream>
#include <iostream>

#include "Layer.h"
#include "Base.h"

Network::Network(
        uint32_t inputSize,
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos,
        const char *path
) :
        m_layerCount(layerCount + 1),
        m_inputSize(inputSize),
        m_outputSize(layerInfos[layerCount - 1].neuronCount),
        m_L(_createLayers(layerCount, layerInfos, path)),
        m_n(_getSizes(layerCount, layerInfos))
{}

Network::~Network()
{
        delete[] m_L;
        delete[] m_n;
}

void Network::encode(
        const char *path
) const {
        // The file must be open in binary mode, and all
        // encode function must write binary.
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw Logger::fatal_error("Error opening file.");

        // Calls the encode function on all layers, the encode
        // function uses std::ofstream::write, writing binary and
        // moving the position of std::ofstream::tellp.
        for (uint32_t L { 0 }; L < m_layerCount - 1; L++)
                m_L[L]->encode(file);

        file.close();
}

const std::unique_ptr<ILayer> *Network::_createLayers(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos
) const {
        const auto layers { new std::unique_ptr<ILayer>[layerCount] };

        layers[0] = Layer::create(m_inputSize, layerInfos[0]);
        for (uint32_t L { 1 }; L < layerCount; L++)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L]);

        return layers;
}

const std::unique_ptr<ILayer> *Network::_createLayers(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos,
        const char *path
) const {
        std::ifstream file(path, std::ios::binary);
        if (!file)
                return _createLayers(layerCount, layerInfos);

        const auto layers { new std::unique_ptr<ILayer>[layerCount] };

        layers[0] = Layer::create(m_inputSize, layerInfos[0], file);
        for (uint32_t L { 1 }; L < layerCount; L++)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L], file);

        file.close();
        return layers;
}

const uint32_t *Network::_getSizes(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos
) const {
        const auto sizes { new uint32_t[layerCount] };

        sizes[0] = m_inputSize;
        for (uint32_t L { 1 }; L < layerCount; L++)
                sizes[L] = layerInfos[L - 1].neuronCount;

        return sizes;
}