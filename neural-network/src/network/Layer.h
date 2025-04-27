#pragma once

#include <memory>

#include <neural-network/network/ILayer.h>

namespace Layer {
        std::unique_ptr<ILayer> create(
                uint32_t                     previousLayerSize,
                const LayerCreateInfo        &layerInfo);

        std::unique_ptr<ILayer> create(
                uint32_t                     previousLayerSize,
                const LayerCreateInfo        &layerInfo,
                std::ifstream                &inputFile);

} // namespace Layer
