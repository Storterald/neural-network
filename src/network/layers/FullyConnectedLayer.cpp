#include "FullyConnectedLayer.h"

#include <random>

#include "../Base.h"

static inline Vector ApplyActivation(FunctionType functionType, const Vector &input) {
        Vector result(input.size());
        switch (functionType) {
                case TANH:
                        if (input.size() < CUDA_MINIMUM)
                                Math::tanh(input.size(), input.data(), result.data());
                        else
                                CudaMath::tanh(input.size(), input.data(), result.data());
                        break;
                case RELU:
                        if (input.size() < CUDA_MINIMUM)
                                Math::ReLU(input.size(), input.data(), result.data());
                        else
                                CudaMath::ReLU(input.size(), input.data(), result.data());
                        break;
                default:
                        throw std::exception("Activation function not implemented.");
        }

        return result;
}

static inline Vector ApplyActivationDerivative(FunctionType functionType, const Vector &input) {
        Vector result(input.size());
        switch (functionType) {
                case TANH:
                        if (input.size() < CUDA_MINIMUM)
                                Math::tanhDerivative(input.size(), input.data(), result.data());
                        else
                                CudaMath::tanhDerivative(input.size(), input.data(), result.data());
                        break;
                case RELU:
                        if (input.size() < CUDA_MINIMUM)
                                Math::ReLUDerivative(input.size(), input.data(), result.data());
                        else
                                CudaMath::ReLUDerivative(input.size(), input.data(), result.data());
                        break;
                default:
                        throw std::exception("Activation function not implemented.");
        }

        return result;
}

FullyConnectedLayer::FullyConnectedLayer(
        uint32_t previousLayerSize,
        uint32_t layerSize,
        FunctionType functionType
)  :
        m_w(previousLayerSize, layerSize),
        // It's way more convenient to have both weights and biases
        // on the same memory. This allows for custom layer kernels.
        m_b(layerSize),
        m_functionType(functionType)
{
        // Initialize the weight matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        for (uint32_t i = 0; i < layerSize; i++)
                for (uint32_t j = 0; j < previousLayerSize; j++)
                        m_w[i][j] = dis(gen);
}

FullyConnectedLayer::FullyConnectedLayer(
        uint32_t previousLayerSize,
        uint32_t layerSize,
        FunctionType functionType,
        std::ifstream &encodedData
) :
        m_w(previousLayerSize, layerSize),
        // Explanation of forceGPU on above constructor.
        m_b(layerSize),
        m_functionType(functionType)
{
        uint32_t sizes[4]{};
        encodedData.read((char *)sizes, 4 * sizeof(uint32_t));

        if (sizes[0] != FULLY_CONNECTED || sizes[1] != functionType)
                throw Logger::fatal_error("Encoded layer infos does not match constructor info.");

        if (previousLayerSize != sizes[2] || layerSize != sizes[3])
                throw Logger::fatal_error("Encoded sizes and constructor sizes parameters do not match.");

        encodedData.read((char *)m_w.data(), m_w.size() * sizeof(float));
        encodedData.read((char *)m_b.data(), m_b.size() * sizeof(float));
}

[[nodiscard]] Vector FullyConnectedLayer::forward(
        const Vector &input
) const {
        //            k=N(L-1)
        // ajL = AFoo(   Σ ak(L-1) * wjkL + bjL)
        //              k=0
        return ApplyActivation(m_functionType, m_w * input + m_b);
}

[[nodiscard]] Vector FullyConnectedLayer::backward(
        // This layer cost
        const Vector &cost,
        // This layer input, aka the previous layer
        // activation values
        const Vector &input
) {
        //  ∂Ce      ∂zjL   ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = 1 * AFoo'(z) * Lcost
        //  ∂bL      ∂bL    ∂zjL   ∂ajL
        const Vector db { ApplyActivationDerivative(m_functionType, m_w * input + m_b) * cost };

        //  ∂Ce      ∂zjL    ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = input * AFoo'(z) * Lcost
        //  ∂wjkL    ∂wjkL   ∂zjL   ∂ajL
        Matrix dw(m_w.width(), m_w.height());

        //   ∂Ce     nL - 1    ∂zjL    ∂ajL    ∂Ce   nL - 1
        // ⎯⎯⎯⎯⎯⎯⎯ =   Σ    ⎯⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ =   Σ    wjkL * AFoo'(z) * Lcost
        // ∂ak(L-1)    j=0   ∂ak(L-1)  ∂zjL   ∂ajL     j=0
        Vector previousCosts(m_w.width());

        if (m_w.size() < CUDA_MINIMUM) {
                // Cycles through the columns of the m_w matrix, this
                // means that it's impossible to use operators on sub-vectors
                // as the columns are not stored contiguously in memory.
                for (uint32_t k { 0 }; k < m_w.width(); k++) {
                        float dCe { 0.0f };
                        for (uint32_t j { 0 }; j < m_w.height(); j++) {
                                dCe += m_w.at(j, k) * db.at(j);
                                dw[{j, k}] = input.at(k) * db.at(j);
                        }

                        previousCosts[k] = dCe;
                }
        } else {
                // Single kernel for the above operations, the unsafe dereference can
                // be used as 'dw' and 'previousCosts' are created with the optional
                // parameter forceGPU
                _backwardGPU(input.data(), dw.data(), db.data(), previousCosts.data());
        }

        {
                // Threads need to be locked when updating shared
                // variables, slows down the program.
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                m_b -= db * LEARNING_RATE;
        }

        return previousCosts;
}

void FullyConnectedLayer::encode(
        std::ofstream &out
) const {
        const uint32_t infos[4] { FULLY_CONNECTED, (uint32_t)m_functionType, m_w.width(), m_w.height() };
        out.write((char *)infos, 4 * sizeof(uint32_t));

        out.write((char *)m_w.data(), m_w.size() * sizeof(float));
        out.write((char *)m_b.data(), m_b.size() * sizeof(float));
}