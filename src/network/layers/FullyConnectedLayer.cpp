#include "FullyConnectedLayer.h"

#include <random>

#include "../../utils/Logger.h"

FullyConnectedLayer::FullyConnectedLayer(
        uint32_t previousLayerSize,
        uint32_t layerSize,
        FunctionType functionType
)  :
        m_w(previousLayerSize, layerSize),
        m_b(layerSize),
        m_functionType(functionType)
{
        // Initialize the weight matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

#ifdef USE_CUDA
        // In CUDA the matrix values are on the GPU memory,
        // so you cannot set it normally
        float *const w { new float[m_w.size()] };
        for (uint32_t i = 0; i < layerSize; i++)
                for (uint32_t j = 0; j < previousLayerSize; j++)
                        w[i * previousLayerSize + j] = dis(gen);

        CUDA_CHECK_ERROR(cudaMemcpy(m_w.data(), w, m_w.size() * sizeof(float), cudaMemcpyHostToDevice),
                "Could copy random values to Layer::m_w.");
        CUDA_CHECK_ERROR(cudaMemset(m_b.data(), 0, layerSize * sizeof(float)),
                "Could not set Layer::m_b memory to 0.");
#else
        for (uint32_t i = 0; i < layerSize; i++)
                for (uint32_t j = 0; j < previousLayerSize; j++)
                        m_w[i][j] = dis(gen);

        std::memset(m_b.data(), 0, m_b.size() * sizeof(float));
#endif // USE_CUDA
}

FullyConnectedLayer::FullyConnectedLayer(
        uint32_t previousLayerSize,
        uint32_t layerSize,
        FunctionType functionType,
        std::ifstream &encodedData
) :
        m_w(previousLayerSize, layerSize),
        m_b(layerSize),
        m_functionType(functionType)
{
        uint32_t sizes[4]{};
        encodedData.read((char *)sizes, 4 * sizeof(uint32_t));

        if (sizes[0] != FULLY_CONNECTED || sizes[1] != functionType)
                throw Logger::fatal_error("Encoded layer infos does not match constructor info.");

        if (previousLayerSize != sizes[2] || layerSize != sizes[3])
                throw Logger::fatal_error("Encoded sizes and constructor sizes parameters do not match.");


#ifdef USE_CUDA
        float *const w { new float[m_w.size()] };
        encodedData.read((char *)w, m_w.size() * sizeof(float));
        CUDA_CHECK_ERROR(cudaMemcpy(m_w.data(), w, m_w.size() * sizeof(float), cudaMemcpyHostToDevice),
                "Could not copy data to GPU.");
        delete[] w;

        float *const b { new float[m_b.size()] };
        encodedData.read((char *)b, m_b.size() * sizeof(float));
        CUDA_CHECK_ERROR(cudaMemcpy(m_b.data(), b, m_b.size() * sizeof(float), cudaMemcpyHostToDevice),
                "Could not copy data to GPU.");
        delete [] b;
#else
        encodedData.read((char *)m_w.data(), m_w.size() * sizeof(float));
        encodedData.read((char *)m_b.data(), m_b.size() * sizeof(float));
#endif // USE_CUDA
}

[[nodiscard]] Vector FullyConnectedLayer::forward(
        const Vector &input
) const {
        //            k=N(L-1)
        // ajL = AFoo(   Σ ak(L-1) * wjkL + bjL)
        //              k=0
#ifdef USE_CUDA
        const Vector z(m_b.size());
        Utils::forward(z.data(), input.data(), m_w.data(), m_b.data(), m_w.width(), m_w.height());
        return ApplyActivation(m_functionType, z);
#else
        return ApplyActivation(m_functionType, m_w * input + m_b);
#endif // USE_CUDA
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
#ifdef USE_CUDA
        const Vector z(m_b.size());
        Utils::forward(z.data(), input.data(), m_w.data(), m_b.data(), m_w.width(), m_w.height());
        const Vector db { ApplyActivationDerivative(m_functionType, z) * cost };
#else
        const Vector db { ApplyActivationDerivative(m_functionType, m_w * input + m_b) * cost };
#endif // USE_CUDA

        //  ∂Ce      ∂zjL    ∂ajL   ∂Ce
        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ = input * AFoo'(z) * Lcost
        //  ∂wjkL    ∂wjkL   ∂zjL   ∂ajL
        Matrix dw(m_w.width(), m_w.height());

        //   ∂Ce     nL - 1    ∂zjL    ∂ajL    ∂Ce   nL - 1
        // ⎯⎯⎯⎯⎯⎯⎯ =   Σ    ⎯⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯ =   Σ    wjkL * AFoo'(z) * Lcost
        // ∂ak(L-1)    j=0   ∂ak(L-1)  ∂zjL   ∂ajL     j=0
        Vector previousCosts(m_w.width());

#ifdef USE_CUDA
        Utils::backward(
                previousCosts.data(), dw.data(), db.data(), m_w.data(),
                input.data(), m_w.width(), m_w.height()
        );

        {
                std::lock_guard lock(m_mutex);

                Utils::fmaScalar(m_w.data(), dw.data(), -LEARNING_RATE, m_w.data(), m_w.size());
                Utils::fmaScalar(m_b.data(), db.data(), -LEARNING_RATE, m_b.data(), m_b.size());
        }
#else
        // Cycles through the columns of the m_w matrix, this
        // means that it's impossible to use operators on sub-vectors
        // as the columns are not stored contiguously in memory.
        for (uint32_t k { 0 }; k < m_w.width(); k++) {
                float dCe { 0.0f };
                for (uint32_t j { 0 }; j < m_w.height(); j++) {
                        dCe += m_w.at(j)[k] * db.at(j);
                        dw[j][k] = input.at(k) * db.at(j);
                }

                previousCosts[k] = dCe;
        }

        {
                // Threads need to be locked when updating shared
                // variables, slows down the program.
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                m_b -= db * LEARNING_RATE;
        }
#endif // USE_CUDA

        return previousCosts;
}

void FullyConnectedLayer::encode(
        std::ofstream &out
) const {
        const uint32_t infos[4] { FULLY_CONNECTED, (uint32_t)m_functionType, m_w.width(), m_w.height() };
        out.write((char *)infos, 4 * sizeof(uint32_t));

#ifdef USE_CUDA
        float *const w { new float[m_w.size()] };
        CUDA_CHECK_ERROR(cudaMemcpy(w, m_w.data(), m_w.size() * sizeof(float), cudaMemcpyDeviceToHost),
                "Could not copy data to CPU.");
        float *const b { new float[m_b.size()] };
        CUDA_CHECK_ERROR(cudaMemcpy(b, m_b.data(), m_b.size() * sizeof(float), cudaMemcpyDeviceToHost),
                "Could not copy data to CPU.");

        out.write((char *)w, m_w.size() * sizeof(float));
        out.write((char *)b, m_b.size() * sizeof(float));

        delete [] w;
        delete [] b;
#else
        out.write((char *)m_w.data(), m_w.size() * sizeof(float));
        out.write((char *)m_b.data(), m_b.size() * sizeof(float));
#endif // USE_CUDA
}