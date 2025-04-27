#pragma once

#include <cstdint>
#include <ostream>
#include <mutex>

#include <neural-network/network/ILayer.h>
#include <neural-network/types/Matrix.h>
#include <neural-network/types/Vector.h>
#include <neural-network/Base.h>

NN_BEGIN

class FullyConnectedLayer final : public ILayer {
public:
        FullyConnectedLayer(
                uint32_t            previousLayerSize,
                uint32_t            layerSize,
                FunctionType        functionType);

        FullyConnectedLayer(
                uint32_t             previousLayerSize,
                uint32_t             layerSize,
                FunctionType         functionType,
                std::ifstream        &encodedData);

        [[nodiscard]] Vector forward(const Vector &input) const override;
        [[nodiscard]] Vector backward(const Vector &cost, const Vector &input) override;

        void encode(std::ostream &out) const override;

        [[nodiscard]] inline uint32_t size() const override
        {
                return m_b.size();
        }

private:
        Matrix                    m_w;
        Vector                    m_b;
        const FunctionType        m_functionType;
        std::mutex                m_mutex;

        void _d_backward(
                const float        input[],
                float              dw[],
                const float        db[],
                float              result[]) const;

}; // class FullyConnectedLayer

NN_END
