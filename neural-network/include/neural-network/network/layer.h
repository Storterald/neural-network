#pragma once

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

#include <cstdint>
#include <memory>

#include <neural-network/types/vector.h>

namespace nn {

enum function_type : uint32_t {
        TANH,
        RELU

}; // enum function_type

enum layer_type : uint32_t {
        FULLY_CONNECTED

}; // enum layer_type

struct layer_create_info {
        layer_type           type;
        function_type        functionType;
        uint32_t             neuronCount;

}; // struct layer_create_info

class layer {
public:
        // TODO templates... maybe?

        static std::unique_ptr<layer> create(
                uint32_t                       previousLayerSize,
                const layer_create_info        &layerInfo);

        static std::unique_ptr<layer> create(
                uint32_t                       previousLayerSize,
                const layer_create_info        &layerInfo,
                std::ifstream                  &inputFile);

#ifdef BUILD_CUDA_SUPPORT
                static std::unique_ptr<layer> create(
                        uint32_t                       previousLayerSize,
                        const layer_create_info        &layerInfo,
                        cudaStream_t                   stream);

                static std::unique_ptr<layer> create(
                        uint32_t                       previousLayerSize,
                        const layer_create_info        &layerInfo,
                        std::ifstream                  &inputFile,
                        cudaStream_t                   stream);
#endif // BUILD_CUDA_SUPPORT

        [[nodiscard]] virtual vector forward(const vector &input) const = 0;
        virtual vector backward(const vector &cost, const vector &input) = 0;

        virtual void encode(std::ostream &file) const = 0;

        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~layer() = default;

}; // interface layer

} // namespace nn
