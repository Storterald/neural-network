#pragma once

#include <cstdint>

#include <neural-network/utils/exceptions.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

class input_layer final : public layer {
public:
        input_layer(uint32_t size) : m_size(size) {}

        [[nodiscard]] inline vector forward(const vector &input) const override
        {
                return input;
        }

        [[nodiscard]] inline vector backward([[maybe_unused]] const vector &cost, [[maybe_unused]] const vector &input) override
        {
                throw fatal_error("Cannot call backward on an input layer");
        }

        void encode([[maybe_unused]] std::ostream &out) const override {}

        [[nodiscard]] inline uint32_t size() const noexcept override
        {
                return m_size;
        }

private:
        uint32_t m_size;

}; // class input_layer

} // namespace nn
