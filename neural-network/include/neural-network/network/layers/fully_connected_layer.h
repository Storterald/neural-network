#pragma once

#include <iostream>
#include <cstdint>
#include <mutex>

#include <neural-network/network/layer.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>

namespace nn {

class fully_connected_layer final : public layer {
public:
        fully_connected_layer(
                uint32_t             previousLayerSize,
                uint32_t             layerSize,
                function_type        functionType);

        fully_connected_layer(
                uint32_t             previousLayerSize,
                uint32_t             layerSize,
                function_type        functionType,
                std::istream         &encodedData);

        [[nodiscard]] vector forward(const vector &input) const override;
        [[nodiscard]] vector backward(const vector &cost, const vector &input) override;

        void encode(std::ostream &out) const override;

        [[nodiscard]] inline uint32_t size() const noexcept override
        {
                return m_b.size();
        }

private:
        matrix                     m_w;
        vector                     m_b;
        const function_type        m_functionType;
        std::mutex                 m_mutex;

        void _d_backward(
                const float        input[],
                float              dw[],
                const float        db[],
                float              result[]) const;

}; // class fully_connected_layer

} // namespace nn
