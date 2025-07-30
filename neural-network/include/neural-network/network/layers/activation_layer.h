#pragma once

#include <cstdint>
#include <limits>

#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

class activation_layer final : public layer {
public:
        activation_layer(function_type activation) : m_activation(activation) {}

        [[nodiscard]] vector forward(const vector &input) const override;
        [[nodiscard]] vector backward(const vector &cost, [[maybe_unused]] const vector &input) override;

        void encode([[maybe_unused]] std::ostream &out) const override {}

        [[nodiscard]] inline uint32_t size() const noexcept override
        {
                return std::numeric_limits<uint32_t>::max();
        }

private:
        function_type m_activation;

}; // class activation_layer

} // namespace nn
