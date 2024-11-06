#pragma once

#include <memory>

#include "ILayer.h"

class Network {
        friend class Train;

public:
        Network() = default;

        Network(
                uint32_t inputSize,
                uint32_t layerCount,
                const LayerCreateInfo layerInfos[],
                const char *path
        );

        ~Network();

        [[nodiscard]] Vector forward(Vector input) const;

        void backward(const Vector &input, const Vector &cost);

        void backward(Vector cost, const Vector activationValues[]);

        void encode(const char *path) const;

        [[nodiscard]] inline uint32_t layerCount() const
        {
                return m_layerCount;
        }

        [[nodiscard]] inline uint32_t inputSize() const
        {
                return m_inputSize;
        }

        [[nodiscard]] inline uint32_t outputSize() const
        {
                return m_outputSize;
        }

private:
        uint32_t m_layerCount { 0 };
        uint32_t m_inputSize { 0 };
        uint32_t m_outputSize { 0 };
        std::unique_ptr<ILayer> *m_L { nullptr };    // [m_layerCount - 1]
        uint32_t *m_n { nullptr };                   // [m_layerCount]

        std::unique_ptr<ILayer> *_createLayers(uint32_t layerCount, const LayerCreateInfo *layerInfos) const;
        std::unique_ptr<ILayer> *_createLayers(uint32_t layerCount, const LayerCreateInfo *layerInfos, const char *path) const;
        uint32_t *_getSizes(uint32_t layerCount, const LayerCreateInfo *layerInfos) const;

}; // class Layer