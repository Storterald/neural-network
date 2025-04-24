#include "FullyConnectedLayer.h"

#include <random>

#include "../../math/Math.h"

static inline Vector _activation(
        FunctionType functionType,
        const Vector &input) {

        Vector result(input.size());
        switch (functionType) {
        case TANH:
                Math::tanh(input.size(), input.data(), result.data());
                break;
        case RELU:
                Math::ReLU(input.size(), input.data(), result.data());
                break;
        default:
                throw LOGGER_EX("Activation function not implemented.");
        }

        return result;
}

static inline Vector _activation_derivative(
        FunctionType functionType,
        const Vector &input) {

        Vector result(input.size());
        switch (functionType) {
        case TANH:
                Math::tanh_derivative(input.size(), input.data(), result.data());
                break;
        case RELU:
                Math::ReLU_derivative(input.size(), input.data(), result.data());
                break;
        default:
                throw LOGGER_EX("Activation function not implemented.");
        }

        return result;
}

FullyConnectedLayer::FullyConnectedLayer(
        uint32_t previousLayerSize,
        uint32_t layerSize,
        FunctionType functionType) :

        m_w(previousLayerSize, layerSize),
        m_b(layerSize),
        m_functionType(functionType) {

        // Initialize the weight matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        for (uint32_t i{}; i < layerSize; i++)
                for (uint32_t j{}; j < previousLayerSize; j++)
                        m_w[i][j] = dis(gen);
}

FullyConnectedLayer::FullyConnectedLayer(
        uint32_t previousLayerSize,
        uint32_t layerSize,
        FunctionType functionType,
        std::ifstream &encodedData) :

        m_w(previousLayerSize, layerSize),
        m_b(layerSize),
        m_functionType(functionType) {

        uint32_t sizes[4]{};
        encodedData.read((char *)sizes, 4 * sizeof(uint32_t));

        if (sizes[0] != FULLY_CONNECTED || sizes[1] != functionType)
                throw LOGGER_EX("Encoded layer infos does not match constructor info.");

        if (previousLayerSize != sizes[2] || layerSize != sizes[3])
                throw LOGGER_EX("Encoded sizes and constructor sizes parameters do not match.");

        encodedData.read((char *)m_w.data(), std::streamsize(m_w.size() * sizeof(float)));
        encodedData.read((char *)m_b.data(), std::streamsize(m_b.size() * sizeof(float)));
}

Vector FullyConnectedLayer::forward(const Vector &input) const
{
        //            k=N(L-1)
        // ajL = AFoo(   Σ ak(L-1) * wjkL + bjL)
        //              k=0
        return _activation(m_functionType, m_w * input + m_b);
}

Vector FullyConnectedLayer::backward(const Vector &cost, const Vector &input)
{
        //  ∂Ce      ∂zjL   ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = 1 * AFoo'(z) * Lcost
        //  ∂bL      ∂bL    ∂zjL   ∂ajL
        const Vector db = _activation_derivative(
                m_functionType, m_w * input + m_b) * cost;

        //  ∂Ce      ∂zjL    ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = input * AFoo'(z) * Lcost
        //  ∂wjkL    ∂wjkL   ∂zjL   ∂ajL
        Matrix dw(m_w.width(), m_w.height());

        //   ∂Ce     nL - 1    ∂zjL    ∂ajL    ∂Ce   nL - 1
        // ⎯⎯⎯⎯⎯⎯⎯ =   Σ    ⎯⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ =   Σ    wjkL * AFoo'(z) * Lcost
        // ∂ak(L-1)    j=0   ∂ak(L-1)  ∂zjL   ∂ajL     j=0
        Vector previousCosts(m_w.width());

#ifdef BUILD_CUDA_SUPPORT
        if (m_w.size() < CUDA_MINIMUM) {
#endif // BUILD_CUDA_SUPPORT
                // Cycles through the columns of the m_w matrix, this
                // means that it's impossible to use operators on sub-vectors
                // as the columns are not stored contiguously in memory.
                for (uint32_t k = 0; k < m_w.width(); ++k) {
                        float dCe = 0.0f;
                        for (uint32_t j = 0; j < m_w.height(); ++j) {
                                dCe += m_w.at(j, k) * db.at(j);
                                dw[{j, k}] = input.at(k) * db.at(j);
                        }

                        previousCosts[k] = dCe;
                }
#ifdef BUILD_CUDA_SUPPORT
        } else {
                // Single kernel for the above operations, the unsafe dereference can
                // be used as 'dw' and 'previousCosts' are created with the optional
                // parameter forceGPU
                _d_backward(input.data(), dw.data(), db.data(), previousCosts.data());
        }
#endif // BUILD_CUDA_SUPPORT

        {
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                m_b -= db * LEARNING_RATE;
        }

        return previousCosts;
}

void FullyConnectedLayer::encode(std::ostream &out) const
{
        const uint32_t infos[4] = {
                FULLY_CONNECTED,
                m_functionType,
                m_w.width(),
                m_w.height()
        };

        out.write((char *)infos, 4 * sizeof(uint32_t));

        out.write((char *)m_w.data(), std::streamsize(m_w.size() * sizeof(float)));
        out.write((char *)m_b.data(), std::streamsize(m_b.size() * sizeof(float)));
}