#pragma once

#include <fstream>
#include <mutex>
#include <random>

#include "ILayer.h"
#ifdef USE_CUDA
#include "../cuda/Matrix.cuh"
#include "../cuda/Utils.cuh"
#include "../cuda/FastMath.cuh"
#else
#include "../math/Matrix.h"
#include "../math/FastMath.h"
#endif

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

        explicit Layer(
                std::ifstream &encodedData
        ) {
                constexpr uint32_t sizes[2]{};
                encodedData.read((char *)sizes, 2 * sizeof(uint32_t));

#ifdef USE_CUDA
                m_w = Matrix(sizes[0], sizes[1]);
                m_b = Vector(sizes[1]);

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
                m_w = Matrix(sizes[0], sizes[1]);
                m_b = Vector(sizes[1]);

                encodedData.read((char *)m_w.data(), m_w.size() * sizeof(float));
                encodedData.read((char *)m_b.data(), m_b.size() * sizeof(float));
#endif // USE_CUDA
        }

        [[nodiscard]] Vector forward(
                const Vector &input
        ) const override {
                //            k=N(L-1)
                // ajL = AFoo(   Σ ak(L-1) * wjkL + bjL)
                //              k=0
#ifdef USE_CUDA
                const Vector z(m_b.size());
                Utils::forward(z.data(), input.data(), m_w.data(), m_b.data(), m_w.width(), m_w.height());
                return ApplyActivation<FunctionType>(z);
#else
                return ApplyActivation<FunctionType>(m_w * input + m_b);
#endif // USE_CUDA
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
#ifdef USE_CUDA
                const Vector z(m_b.size());
                Utils::forward(z.data(), input.data(), m_w.data(), m_b.data(), m_w.width(), m_w.height());
                const Vector db { ApplyActivationDerivative<FunctionType>(z) * cost };
#else
                const Vector db { ApplyActivationDerivative<FunctionType>(m_w * input + m_b) * cost };
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
#endif // USE_CUDA

                return previousCosts;
        }

        void encode(
                std::ofstream &out
        ) const override {
                const uint32_t sizes[2] { m_w.width(), m_w.height() };
                out.write((char *)sizes, 2 * sizeof(uint32_t));

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

        uint32_t size() const override
        {
                return m_b.size();
        }

private:
        Matrix m_w;
        Vector m_b;
        mutable std::mutex m_mutex;

}; // class Layer