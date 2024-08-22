#pragma once

#include <cstring>
#include <fstream>
#include <mutex>
#include <random>

#include "../math/Matrix.h"
#include "../enums/FunctionType.h"
#include "../enums/LayerType.h"
#include "../math/FastMath.h"

struct LayerCreateInfo {
        LayerType type;
        FunctionType functionType;
        uint32_t neuronCount;
};

template<FunctionType FunctionType>
Vector ApplyActivation(const Vector& input) {
        CONSTEXPR_SWITCH(FunctionType,
                CASE(TANH, return Fast::tanh(input) ),
                CASE(RELU, return Fast::relu(input) )
        );

        // Unreachable, the switch above should cover all possible
        // options present in the FunctionType enum.
        return {};
}

template<FunctionType FunctionType>
Vector ApplyActivationDerivative(const Vector& input) {
        CONSTEXPR_SWITCH(FunctionType,
                CASE(TANH, return Fast::tanhDerivative(input) ),
                CASE(RELU, return Fast::reluDerivative(input) )
        );

        // Unreachable, the switch above should cover all possible
        // options present in the FunctionType enum.
        return {};
}

// The interface ILayer, all layers inherit this.
class ILayer {
public:
        [[nodiscard]] virtual Vector forward(const Vector &input) const = 0;
        virtual Vector backward(const Vector &cost, const Vector &input) = 0;

        virtual void encode(std::ofstream &) const = 0;
        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~ILayer() = default;

}; // interface ILayer

// Generic implementation of the Layer class
template<LayerType, FunctionType> class Layer : public ILayer {
        Layer(uint32_t previousLayerSize, uint32_t layerSize);
        Layer(std::ifstream &encodedData);
};

// Specialized implementation of the Layer class
template<FunctionType FunctionType>
class Layer<FULLY_CONNECTED, FunctionType> final : public ILayer {
public:
        Layer(
                uint32_t previousLayerSize, uint32_t layerSize
        )  :
                m_w(previousLayerSize, layerSize),
                m_b(layerSize)
        {
                // Initialize the weight matrix with random values
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution dis(-0.5f, 0.5f);

                for (uint32_t i = 0; i < m_w.height(); i++)
                        for (uint32_t j = 0; j < m_w.width(); j++)
                                m_w[i][j] = dis(gen);

                std::memset(m_b.data(), 0, m_b.size() * sizeof(float));
        }

        Layer(
                std::ifstream &encodedData
        ) {
                constexpr uint32_t sizes[2]{};
                encodedData.read((char *)sizes, 2 * sizeof(uint32_t));

                m_w = Matrix(sizes[0], sizes[1]);
                encodedData.read((char *)m_w.data(), m_w.size() * sizeof(float));

                m_b = Vector(sizes[1]);
                encodedData.read((char *)m_b.data(), m_b.size() * sizeof(float));
        }

        [[nodiscard]] Vector forward(
                const Vector &input
        ) const override {
                //            k=N(L-1)
                // ajL = AFoo(   Σ ak(L-1) * wjkL + bjL)
                //              k=0
                return ApplyActivation<FunctionType>(m_w * input + m_b);
        }

        [[nodiscard]] Vector backward(
                // This layer cost
                const Vector &cost,
                // This layer input, also the previous layer
                // activation values
                const Vector &input
        ) override {
                //  ∂Ce      ∂zjL   ∂ajL   ∂Ce
                // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = 1 * AFoo'(z) * Lcost
                //  ∂bL      ∂bL    ∂zjL   ∂ajL
                const Vector db { ApplyActivationDerivative<FunctionType>(m_w * input + m_b) * cost };

                //  ∂Ce      ∂zjL    ∂ajL   ∂Ce
                // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = input * AFoo'(z) * Lcost
                //  ∂wjkL    ∂wjkL   ∂zjL   ∂ajL
                Matrix dw(m_w.width(), m_w.height());

                //   ∂Ce     nL - 1    ∂zjL    ∂ajL    ∂Ce   nL - 1
                // ⎯⎯⎯⎯⎯⎯⎯ =   Σ    ⎯⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ =   Σ    wjkL * AFoo'(z) * Lcost
                // ∂ak(L-1)    j=0   ∂ak(L-1)  ∂zjL   ∂ajL     j=0
                Vector previousCosts(m_w.width());

                for (uint32_t k { 0 }; k < m_w.width(); k++) {
                        float dCe { 0.0f };
                        for (uint32_t j { 0 }; j < m_w.height(); j++) {
                                dCe += m_w.at(j)[k] * db.at(j);
                                dw[j][k] = input.at(k) * db.at(j);
                        }

                        previousCosts[k] = dCe;
                }

                {
                        std::lock_guard lock(m_mutex);
                        m_w -= dw * LEARNING_RATE;
                        m_b -= db * LEARNING_RATE;
                }

                return previousCosts;
        }

        void encode(
                std::ofstream &out
        ) const override {
                const uint32_t sizes[2] { m_w.width(), m_w.height() };
                out.write((char *)sizes, 2 * sizeof(uint32_t));
                out.write((char *)m_w.data(), m_w.size() * sizeof(float));
                out.write((char *)m_b.data(), m_b.size() * sizeof(float));
        }

        uint32_t size() const override
        {
                return m_b.size();
        }

private:
        Matrix m_w;
        Vector m_b;
        mutable std::mutex m_mutex;

}; // class Layer