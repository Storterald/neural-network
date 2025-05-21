#pragma once

#ifdef BUILD_CUDA_SUPPORT
#include <driver_types.h> // cudaStream_t
#endif // BUILD_CUDA_SUPPORT

#include <iostream>
#include <cstdint>
#include <mutex>

#include <neural-network/network/layer.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

class fully_connected_layer final : public layer {
public:
        fully_connected_layer(
                uint32_t             previousLayerSize,
                uint32_t             layerSize,
                function_type        functionType,
                stream               stream = invalid_stream);

        fully_connected_layer(
                uint32_t             previousLayerSize,
                uint32_t             layerSize,
                function_type        functionType,
                std::istream         &encodedData,
                stream               stream = invalid_stream);

        [[nodiscard]] vector forward(const vector &input) const override;
        [[nodiscard]] vector backward(const vector &cost, const vector &input) override;

        void encode(std::ostream &out) const override;

        [[nodiscard]] inline uint32_t size() const noexcept override
        {
                return m_b.size();
        }

private:
#ifdef BUILD_CUDA_SUPPORT
        stream                         m_stream;
#else
        static constexpr stream        m_stream = invalid_stream;
#endif // BUILD_CUDA_SUPPORT
        matrix                         m_w;
        vector                         m_b;
        const function_type            m_functionType;
        std::mutex                     m_mutex;

        void _d_backward(
                const float        input[],
                float              dw[],
                const float        db[],
                float              result[]) const;

}; // class fully_connected_layer

} // namespace nn
