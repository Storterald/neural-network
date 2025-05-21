#include <neural-network/network/layers/fully_connected_layer.h>

#include <cstdint>
#include <random>
#include <mutex>

#include <neural-network/utils/exceptions.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/buf.h>
#include <neural-network/math/math.h>
#include <neural-network/base.h>

static nn::vector _activation(
        nn::function_type        functionType,
        const nn::vector         &input) {

        nn::vector result(input.size(), input.stream());
        switch (functionType) {
        case nn::TANH:
                nn::math::tanh(input.size(), input, result);
                break;
        case nn::RELU:
                nn::math::ReLU(input.size(), input, result);
                break;
        default:
                throw nn::fatal_error("Activation function not implemented.");
        }

        return result;
}

static nn::vector _activation_derivative(
        nn::function_type        functionType,
        const nn::vector         &input) {

        nn::vector result(input.size(), input.stream());
        switch (functionType) {
        case nn::TANH:
                nn::math::tanh_derivative(input.size(), input, result);
                break;
        case nn::RELU:
                nn::math::ReLU_derivative(input.size(), input, result);
                break;
        default:
                throw nn::fatal_error("Activation function not implemented.");
        }

        return result;
}

namespace nn {

fully_connected_layer::fully_connected_layer(
        uint32_t             previousLayerSize,
        uint32_t             layerSize,
        function_type        functionType,
        stream               stream) :

        m_w(previousLayerSize, layerSize, stream),
        m_b(layerSize, stream),
        m_functionType(functionType)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(stream)
#endif // BUILD_CUDA_SUPPORT
{

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        span tmp = m_w.data(buf::HOST, true);
        for (uint32_t i = 0; i < layerSize; i++)
                for (uint32_t j = 0; j < previousLayerSize; j++)
                        tmp[i * m_w.width() + j] = dis(gen);
}

fully_connected_layer::fully_connected_layer(
        uint32_t             previousLayerSize,
        uint32_t             layerSize,
        function_type        functionType,
        std::istream         &encodedData,
        stream               stream) :

        m_w(previousLayerSize, layerSize, stream),
        m_b(layerSize, stream),
        m_functionType(functionType)
#ifdef BUILD_CUDA_SUPPORT
        , m_stream(stream)
#endif // BUILD_CUDA_SUPPORT
{

        uint32_t sizes[4]{};
        encodedData.read((char *)sizes, 4 * sizeof(uint32_t));

        if (sizes[0] != FULLY_CONNECTED || sizes[1] != functionType)
                throw fatal_error("Encoded layer infos does not match constructor info.");

        if (previousLayerSize != sizes[2] || layerSize != sizes[3])
                throw fatal_error("Encoded sizes and constructor sizes parameters do not match.");

        encodedData.read(
                (char *)(float *)m_w.data(buf::HOST, true),
                std::streamsize(m_w.size() * sizeof(float)));
        encodedData.read(
                (char *)(float *)m_b.data(buf::HOST, true),
                std::streamsize(m_b.size() * sizeof(float)));
}

vector fully_connected_layer::forward(const vector &input) const
{
        //            k=N(L-1)
        // ajL = AFoo(   Σ ak(L-1) * wjkL + bjL)
        //              k=0
        return _activation(m_functionType, m_w * input + m_b);
}

vector fully_connected_layer::backward(const vector &cost, const vector &input)
{
        //  ∂Ce      ∂zjL   ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = 1 * AFoo'(z) * Lcost
        //  ∂bL      ∂bL    ∂zjL   ∂ajL
        vector db = _activation_derivative(m_functionType, m_w * input + m_b) * cost;

        //  ∂Ce      ∂zjL    ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = input * AFoo'(z) * Lcost
        //  ∂wjkL    ∂wjkL   ∂zjL   ∂ajL
        matrix dw(m_w.width(), m_w.height(), m_stream);

        //   ∂Ce     nL - 1    ∂zjL    ∂ajL    ∂Ce   nL - 1
        // ⎯⎯⎯⎯⎯⎯⎯ =   Σ    ⎯⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ =   Σ    wjkL * AFoo'(z) * Lcost
        // ∂ak(L-1)    j=0   ∂ak(L-1)  ∂zjL   ∂ajL     j=0
        vector prev(m_w.width(), m_stream);

#ifdef BUILD_CUDA_SUPPORT
        if (m_w.location() == buf::HOST) {
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

                        prev[k] = dCe;
                }
#ifdef BUILD_CUDA_SUPPORT
        } else {
                _d_backward(
                        input.view(buf::DEVICE), dw.data(buf::DEVICE, true),
                        db.view(buf::DEVICE), prev.data(buf::DEVICE, true));
        }
#endif // BUILD_CUDA_SUPPORT

        {
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                m_b -= db * LEARNING_RATE;
        }

        return prev;
}

void fully_connected_layer::encode(std::ostream &out) const
{
        const uint32_t infos[4] = {
                FULLY_CONNECTED,
                m_functionType,
                m_w.width(),
                m_w.height()
        };

        out.write((char *)infos, 4 * sizeof(uint32_t));

        out.write(
                (char *)(const float *)m_w.view(buf::HOST),
                std::streamsize(m_w.size() * sizeof(float)));
        out.write(
                (char *)(const float *)m_b.view(buf::HOST),
                std::streamsize(m_b.size() * sizeof(float)));
}

} // namespace nn
