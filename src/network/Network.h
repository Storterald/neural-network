#pragma once

#include <memory>

#include "ILayer.h"

class Network {
        friend struct Train;

public:
        Network(
                uint32_t inputSize,
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos,
                const char *path
        );

        ~Network();

        inline Vector compute(
                const float *inputs
        ) const {
                Vector aL(m_n[0], inputs);

                for (uint32_t L { 0 }; L < m_layerCount - 1; L++)
                        aL = m_L[L]->forward(aL);

                return aL;
        }

        void encode(
                const char *path
        ) const;

private:
        const uint32_t m_layerCount;
        const uint32_t m_inputSize;
        const uint32_t m_outputSize;

        const std::unique_ptr<ILayer> *m_L;     // [m_layerCount - 1]
        const uint32_t *m_n;                    // [m_layerCount]

        const std::unique_ptr<ILayer> *_createLayers(
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos
        ) const;

        const std::unique_ptr<ILayer> *_createLayers(
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos,
                const char *path
        ) const;

        const uint32_t *_getSizes(
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos
        ) const;

        void _backpropagate(
                const float *inputs,
                const float *y
        );

}; // class Layer