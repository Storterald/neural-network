#pragma once

#include <iostream>
#include <cstdint>
#include <mutex>

#include <neural-network/network/layer.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

class dense_layer final : public layer {
public:
        dense_layer(
                uint32_t             prev,
                uint32_t             size,
                stream               stream = invalid_stream);

        dense_layer(
                uint32_t             prev,
                uint32_t             size,
                std::istream         &in,
                stream               stream = invalid_stream);

        [[nodiscard]] vector forward(const vector &input) const override;
        [[nodiscard]] vector backward(const vector &cost, const vector &input) override;

        void encode(std::ostream &out) const override;

        [[nodiscard]] inline uint32_t size() const noexcept override
        {
                return m_b.size();
        }

private:
        stream     m_stream;
        matrix     m_w;
        vector     m_b;
        std::mutex m_mutex;

        inline void _cpu_backward(
                const float        input[],
                float              dw[],
                const float        db[],
                float              result[]) const;

        void _gpu_backward(
                const float        input[],
                float              dw[],
                const float        db[],
                float              result[]) const;

}; // class dense_layer

} // namespace nn
