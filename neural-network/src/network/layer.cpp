#include <neural-network/network/layer.h>

#include <istream>
#include <cstdint>
#include <memory>

#include <neural-network/network/layers/dense_layer.h>
#include <neural-network/utils/exceptions.h>
#include <neural-network/base.h>

#include "neural-network/network/layers/activation_layer.h"
#include "neural-network/network/layers/input_layer.h"

namespace nn {

std::unique_ptr<layer> layer::create(
        uint32_t                       prev,
        const layer_create_info        &info,
        stream                         stream) {

        switch (info.type) {
        case input:
                return std::make_unique<input_layer>(info.count);
        case dense:
                return std::make_unique<dense_layer>(prev, info.count, stream);
        case activation:
                return std::make_unique<activation_layer>(info.activation);
        default:
                throw fatal_error("Layer type not recognized.");
        }
}

std::unique_ptr<layer> layer::create(
        uint32_t                       prev,
        const layer_create_info        &info,
        std::istream                   &in,
        stream                         stream) {

        switch (info.type) {
        case input:
                return std::make_unique<input_layer>(info.count);
        case dense:
                return std::make_unique<dense_layer>(prev, info.count, in, stream);
        case activation:
                return std::make_unique<activation_layer>(info.activation);
        default:
                throw fatal_error("Layer type not recognized.");
        }
}

} // namespace nn
