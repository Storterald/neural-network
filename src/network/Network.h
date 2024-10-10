#pragma once

#include <memory>

#include "ILayer.h"

class Network {
        friend class Train;

public:
        const uint32_t m_layerCount;
        const uint32_t m_inputSize;
        const uint32_t m_outputSize;

        Network(
                uint32_t inputSize,
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos,
                const char *path
        );

        ~Network();

        [[nodiscard]] inline Vector forward(
                Vector aL
        ) const {
                for (uint32_t L { 0 }; L < m_layerCount - 1; L++)
                        aL = m_L[L]->forward(aL);

                return aL;
        }

        inline void backward(
                Vector dC,
                const Vector *a
        ) {
                for (int32_t L { (int32_t)m_layerCount - 2 }; L >= 0; L--)
                        dC = m_L[L]->backward(dC, a[L]);
        }

        void encode(
                const char *path
        ) const;

private:


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

}; // class Layer