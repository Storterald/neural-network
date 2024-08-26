#include "Network.h"

#include "Layer.h"
#include "../utils/Logger.h"

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

const std::unique_ptr<ILayer> *m_L;     // [m_layerCount - 1]
const uint32_t *m_n;                    // [m_layerCount]

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

void Network::_backpropagate(
        const float *inputs,
        const float *y
) {
        const auto a { new Vector[m_layerCount] };
        a[0] = Vector(m_inputSize, inputs);

        for (uint32_t L { 1 }; L < m_layerCount; L++)
                a[L] = m_L[L - 1]->forward(a[L - 1]);

        // The cost of the last layer neurons is calculated with (ajL - yj) ^ 2,
        // this mean that the derivative is equal to 2 * (ajL - y). ∂C does not need
        // to be an array, as the old values are not needed.
#ifdef USE_CUDA
        Vector dC { (a[m_layerCount - 1] - Vector(m_outputSize, y)) * 2.0f };
#else
        Vector dC { (a[m_layerCount - 1] - y) * 2.0f };
#endif // USE_CUDA

#ifdef DEBUG_MODE_ENABLED
        for (uint32_t L { m_layerCount - 1 }, j { 0 }; j < m_outputSize; j++)
                Log << Logger::pref() << "Cost for output neuron [" << j << "] has value of: " << std::pow(a[L].at(j) - y[j], 2.0f)
                    << ", and it's derivative has value of: " << dC.at(j) << ". Formulas: (" << a[L].at(j) << " - " << y[j]
                    << ")^2, 2 * (" << a[L].at(j) << " - " << y[j] << ").\n";
#endif // DEBUG_MODE_ENABLED

        for (int32_t L { (int32_t)m_layerCount - 2 }; L >= 0; L--)
                dC = m_L[L]->backward(dC, a[L]);

        delete [] a;
}