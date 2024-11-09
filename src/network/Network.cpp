#include "Network.h"

#include <fstream>
#include <filesystem>

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

Vector Network::forward(
        // Not passing as const ref to avoid creating an extra Vector
        // to store the current activation values.
        Vector aL
) const {
        for (uint32_t L { 0 }; L < m_layerCount - 1; L++)
                aL = m_L[L]->forward(aL);

        return aL;
}

void Network::backward(
        const Vector &input,
        const Vector &dC
) {
        Vector *a { new Vector[m_layerCount] };
        a[0] = input;

        for (uint32_t L { 1 }; L < m_layerCount; L++)
                a[L] = m_L[L - 1]->forward(a[L - 1]);

        this->backward(dC, a);

        delete [] a;
}


void Network::backward(
        Vector dC,
        const Vector a[]
) {
        for (int32_t L { (int32_t)m_layerCount - 2 }; L >= 0; L--)
                dC = m_L[L]->backward(dC, a[L]);
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

std::unique_ptr<ILayer> *Network::_createLayers(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos
) const {
        const auto layers { new std::unique_ptr<ILayer>[layerCount] };

        layers[0] = Layer::create(m_inputSize, layerInfos[0]);
        for (uint32_t L { 1 }; L < layerCount; L++)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L]);

        return layers;
}

std::unique_ptr<ILayer> *Network::_createLayers(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos,
        const char *path
) const {
        if (layerCount == 0)
                throw Logger::fatal_error("Cannot initialize Network with no layers. The minimum required "
                                          "amount is 1, the output layer.");

        // If the path is empty use other layer constructor.
        if (path[0] == '\0')
                return _createLayers(layerCount, layerInfos);

        if (!std::filesystem::exists(path))
                throw Logger::fatal_error("File given to Network() does not exist.");

        std::ifstream file(path, std::ios::binary);
        const auto layers { new std::unique_ptr<ILayer>[layerCount] };

        layers[0] = Layer::create(m_inputSize, layerInfos[0], file);
        for (uint32_t L { 1 }; L < layerCount; L++)
                layers[L] = Layer::create(layerInfos[L - 1].neuronCount, layerInfos[L], file);

        file.close();
        return layers;
}

uint32_t *Network::_getSizes(
        uint32_t layerCount,
        const LayerCreateInfo *layerInfos
) const {
        uint32_t *sizes { new uint32_t[layerCount] };

        sizes[0] = m_inputSize;
        for (uint32_t L { 1 }; L < layerCount; L++)
                sizes[L] = layerInfos[L - 1].neuronCount;

        return sizes;
}
