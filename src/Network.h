#pragma once

#include <fstream>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <thread>
#include <barrier>
#include <vector>

#include "math/Matrix.h"
#include "utils/FastMath.h"
#include "utils/Logger.h"
#include "utils/MultiUseBuffer.h"

enum FunctionType {
        TANH,
        RELU
};

// While it would have been nice to have a completely dynamic
// network, it's way easier to have the layer sizes as a variadic
// argument.
template<FunctionType type, uint32_t ...sizes>
class Network {
        static_assert(sizeof...(sizes) > 1, "\n\n"
                "The layer count must be at least 2, the input and the output.");

private:
        // Template dependant compile time constants. Only possible
        // thanks to layer sizes as variadic arguments.
        static constexpr uint32_t LAYER_COUNT { sizeof...(sizes) };
        static constexpr uint32_t NEURONS_COUNT { (0 + ... + sizes) };
        static constexpr uint32_t WEIGHTS_COUNT { [] -> uint32_t {
                uint32_t count { 0 };
                for (uint32_t tmp[] { sizes... }, i { 0 }; i < LAYER_COUNT - 1; i++)
                        count += tmp[i] * tmp[i+1];

                return count;
        }() };
        static constexpr uint32_t BIASES_COUNT { [] -> uint32_t {
                uint32_t count { 0 };
                for (uint32_t tmp[] { sizes... }, i { 1 }; i < LAYER_COUNT; i++)
                        count += tmp[i];

                return count;
        }() };

        MultiUseBuffer<
                LAYER_COUNT * sizeof(Vector)
        > m_buffer{};

        // Nomenclature from 3Blue1Brown.
        // w is an array containing the weights, b contains the biases, n contains the sizes,
        // a contains the activation value, and z contains the pre non-linear function
        // activation value.
        static constexpr uint32_t s_n[sizeof...(sizes)] { sizes... };
        Matrix m_w[LAYER_COUNT]{};
        Vector m_b[LAYER_COUNT]{};

        Vector _getInfluence(
                const float *inputs,
                const float *y
        ) {
                constexpr uint32_t INPUT_NEURONS { s_n[0] };
                constexpr uint32_t OUTPUT_NEURONS { s_n[LAYER_COUNT - 1] };

                Vector z[LAYER_COUNT]{}, a[LAYER_COUNT]{};
                for (uintptr_t L { 0 }; L < LAYER_COUNT; L++) {
                        z[L] = Vector(s_n[L]);
                        a[L] = Vector(s_n[L]);
                }

                std::memcpy(a[0].data(), inputs, INPUT_NEURONS * sizeof(float));

                // The compute function saving the pre non-linear function values.
                for (uint32_t L { 1 }; L < LAYER_COUNT; L++) {
                        z[L] = m_w[L] * a[L - 1] + m_b[L];

                        CONSTEXPR_SWITCH(type,
                                CASE(TANH, a[L] = Fast::tanh(z[L])),
                                CASE(RELU, a[L] = Fast::relu(z[L]))
                        );
                }

                // An array containing the derivatives of the costs of all neurons.
                // The array has an "inverted" structure, with the neurons from the last layer first,
                // and the neurons from the first layer last. This allows for easier access and avoidance
                // of offsets during computation.
                float costsDerivatives[NEURONS_COUNT]{};

                // The cost of the last layer neurons is calculated with (ajL - yj) ^ 2,
                // this mean that the derivative is equal to 2 * (ajL - y)
                Vector outputCosts { (a[LAYER_COUNT - 1] - y) * 2.0f };
                std::memcpy(costsDerivatives, outputCosts.data(), OUTPUT_NEURONS * sizeof(float));
#ifdef DEBUG_MODE_ENABLED
                for (uint32_t L { LAYER_COUNT - 1 }, j { 0 }; j < OUTPUT_NEURONS; j++)
                        Log << Logger::pref() << "Cost for output neuron [" << j << "] has value of: " << std::pow(a[L][j] - y[j], 2.0f)
                            << ", and it's derivative has value of: " << outputCosts[j] << ". Formulas: (" << a[L][j] << " - " << y[j]
                            << ")^2, 2 * (" << a[L][j] << " - " << y[j] << ").\n";
#endif // DEBUG_MODE_ENABLED

                // If all costs are less then 'EPSILON', the function returns all 0s.
                if (std::all_of(costsDerivatives, costsDerivatives + OUTPUT_NEURONS, [](float v) -> bool { return std::abs(v) < EPSILON; }))
                        return Vector(WEIGHTS_COUNT + BIASES_COUNT);

                // Weights and biases influences
                Vector result(WEIGHTS_COUNT + BIASES_COUNT);
                float *const dw { result.data() }, *const db { result.data() + WEIGHTS_COUNT };

                // Cycle trough all layers 'L', reading layer 'L' and 'L-1' going backward
                for (uint32_t costOffset { 0 }, weightOffset { WEIGHTS_COUNT }, biasesOffset { BIASES_COUNT }, L { LAYER_COUNT - 1 }; L >= 1; L--) {
                        // Since the loop is going backward and changing 2 layers at a time,
                        // the 'L' layer costs have already been calculated in the previous
                        // iteration or when initializing the cost derivatives.
                        const uint32_t nL { s_n[L] };
                        const float *const aLCosts { costsDerivatives + costOffset };

                        const uint32_t nL1 { s_n[L - 1] };
                        float *const aL1Costs { (float *)aLCosts + nL };

                        costOffset += nL;
                        weightOffset -= m_w[L].size();
                        biasesOffset -= m_b[L].size();

                        //  ∂Ce      ∂zjL   ∂ajL   ∂Ce
                        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯
                        //  ∂bL      ∂bL    ∂zjL   ∂ajL
                        Vector dbL{};
                        CONSTEXPR_SWITCH(type,
                                // Avoid extra calculations using the already existing tanh values stored in a[L]
                                CASE(TANH, dbL = Fast::tanhDerivativeFromTanh(a[L]) * aLCosts),
                                CASE(RELU, dbL = Fast::reluDerivative(z[L]) * aLCosts)
                        );
                        std::memcpy(db + biasesOffset, dbL.data(), nL * sizeof(float));

                        // The backward step cycles in the opposite way as the forward step,
                        // it cycles 'k' times 'j' times, instead of 'j' times 'k' times.
                        // Unfortunately this slows things down since the matrices are cycled
                        // through based on columns not rows.
                        for (uint32_t k { 0 }; k < nL1; k++) {
                                //   ∂Ce     nL - 1    ∂zjL    ∂ajL    ∂Ce
                                // ⎯⎯⎯⎯⎯⎯⎯ =   Σ    ⎯⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯
                                // ∂ak(L-1)    j=0   ∂ak(L-1)  ∂zjL   ∂ajL
                                float dCe { 0 };

                                for (uint32_t j { 0 }; j < nL; j++) {
                                        // Not computing dbL and the non linear derivatives here since it would
                                        // be computed 'k' more times than necessary, with the same output.

                                        dCe += m_w[L][j][k] * dbL[j];

                                        //  ∂Ce      ∂zjL    ∂ajL   ∂Ce
                                        // ⎯⎯⎯⎯⎯ = ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯⎯ ⎯⎯⎯⎯⎯
                                        //  ∂wjkL    ∂wjkL   ∂zjL   ∂ajL
                                        dw[j * nL1 + k + weightOffset] = a[L - 1][k] * dbL[j];
                                }

                                aL1Costs[k] = dCe;
                        }
                }

                return result;
        }

        Vector _dispatchWorkers(
                uint32_t count,
                const float *inputs,
                const float *outputs
        ) {
                constexpr uint32_t INPUT_NEURONS { s_n[0] };
                constexpr uint32_t OUTPUT_NEURONS { s_n[LAYER_COUNT - 1] };
#ifdef DEBUG_MODE_ENABLED
                const uint32_t THREAD_COUNT { 1 };
#else
                const uint32_t THREAD_COUNT { std::thread::hardware_concurrency() };
#endif // DEBUG_MODE_ENABLED
                const uint32_t CHUNK_SIZE { count / THREAD_COUNT };
                const uint32_t REMAINDER { count % THREAD_COUNT };

                std::barrier sync(THREAD_COUNT);
                std::vector<Vector> results(THREAD_COUNT, Vector(WEIGHTS_COUNT + BIASES_COUNT));
                
                auto worker = [&](uint32_t start, uint32_t end, uint32_t thread) {
                        for (uint32_t i { start }; i < end; i++)
                                results[thread] += _getInfluence(inputs + i * INPUT_NEURONS, outputs + i * OUTPUT_NEURONS);

                        sync.arrive_and_wait();
                };

                std::vector<std::jthread> threads(THREAD_COUNT);
                for (uint32_t start { 0 }, i { 0 }; i < THREAD_COUNT; i++) {
                        uint32_t end { start + CHUNK_SIZE + (i < REMAINDER ? 1 : 0) };
                        threads[i] = std::jthread(worker, start, end, i);
                        start = end;
                }

                Vector average(WEIGHTS_COUNT + BIASES_COUNT);
                for (uint32_t i { 0 }; i < THREAD_COUNT; i++)
                        average += results[i];

                average /= -(float)count;

                return average;
        }

public:
        Network(
                const float *weights,
                const float *biases
        ) {
                // Initializing compute buffer index 0
                Vector *const computeBuffer { (Vector *)m_buffer.template get<0>() };
                computeBuffer[0] = Vector(s_n[0]);

                // Initialize all weight matrices and bias vectors
                for (uint32_t weightOffset { 0 }, biasesOffset { 0 }, L { 1 }; L < LAYER_COUNT; L++) {
                        const uint32_t nL { s_n[L] };
                        const uint32_t nL1 { s_n[L - 1] };

                        m_w[L] = Matrix(nL1, nL);
                        std::memcpy(m_w[L].data(), weights + weightOffset, m_w[L].size() * sizeof(float));
                        weightOffset += m_w[L].size();

                        m_b[L] = Vector(nL);
                        std::memcpy(m_b[L].data(), biases + biasesOffset, m_b[L].size() * sizeof(float));
                        biasesOffset += m_b[L].size();

                        // Initializing buffer of empty vectors
                        computeBuffer[L] = Vector(s_n[L]);
                }
        }

        Vector compute(
                const float *inputs
        ) {
                constexpr uint32_t INPUT_NEURONS { s_n[0] };

                Vector *const a { (Vector *)m_buffer.template get<0>() };

                // Copying inputs to a0.
                std::memcpy(a[0].data(), inputs, INPUT_NEURONS * sizeof(float));

                // Cycle trough all layers 'L', reading layer 'L' and 'L-1' going forward
                for (uint32_t L { 1 }; L < LAYER_COUNT; L++) {
                        // For every neuron activation 'a' in layer 'L' at index 'j', the value is:
                        //          k=N(L-1)
                        // ajL = tanh( Σ ak(L-1) * wjkL) + bjL
                        //            k=0
                        //
                        // This can be converted in a simpler expression with matrices and vectors.
                        CONSTEXPR_SWITCH(type,
                                CASE(TANH, a[L] = Fast::tanh(m_w[L] * a[L - 1] + m_b[L])),
                                CASE(RELU, a[L] = Fast::relu(m_w[L] * a[L - 1] + m_b[L]))
                        );
                }

                return a[LAYER_COUNT - 1];
        }

        void trainWithBackpropagation(
                uint32_t sampleSize,
                const float *pInputs,
                const float *pOutputs
        ) {
                // Gets the influences using all threads.
                Vector negativeGradientVector { _dispatchWorkers(sampleSize, pInputs, pOutputs) };

                // Since the values are in a single vector of size WEIGHTS_COUNT + BIASES_COUNT,
                // the vector is split in 2 and then split in the sub-matrices and sub-vectors it contains.
                const float *const weightsData { negativeGradientVector.data() };
                const float *const biasesData { negativeGradientVector.data() + WEIGHTS_COUNT };

#ifdef DEBUG_MODE_ENABLED
                Log << Logger::pref() << "Internal values changes:\n";
#endif // DEBUG_MODE_ENABLED
                // Changing the weights and biases by the negative of their
                // influence times the learning rate.
                for (uint32_t wOffset { 0 }, bOffset { 0 }, L { 1 }; L < LAYER_COUNT; L++) {
                        // Creating a matrix with the data array to use the
                        // custom defined operators. The operators are likely
                        // way faster since they use AVX-512.
                        Matrix negativeWeightsInfluence(s_n[L - 1], s_n[L]);
                        std::memcpy(negativeWeightsInfluence.data(), weightsData + wOffset, negativeWeightsInfluence.size() * sizeof(float));
                        wOffset += negativeWeightsInfluence.size();

#ifdef DEBUG_MODE_ENABLED
                        for (uint32_t i { 0 }; i < m_w[L].size(); i++)
                                Log << Logger::pref() << "Change of weight wj" << i / m_w[L].width() << "k" << i % m_w[L].width()
                                    << "L"<< L << " has a changed from: " << std::setprecision(8) << m_w[L].data()[i] << " to: "
                                    << std::setprecision(8) << m_w[L].data()[i] + negativeWeightsInfluence.data()[i] * LEARNING_RATE<< ".\n";
#endif // DEBUG_MODE_ENABLED

                        m_w[L] += negativeWeightsInfluence * LEARNING_RATE;

                        // As explained above, creating a vector with the
                        // data array to use the fast operators.
                        Vector negativeBiasesInfluence(s_n[L]);
                        std::memcpy(negativeBiasesInfluence.data(), biasesData + bOffset, negativeBiasesInfluence.size() * sizeof(float));
                        bOffset += negativeBiasesInfluence.size();

#ifdef DEBUG_MODE_ENABLED
                        for (uint32_t i { 0 }; i < m_b[L].size(); i++)
                                Log << Logger::pref() << "Change of bias bj" << i << "L" << L
                                    << " has a changed from: " << std::setprecision(8) << m_b[L][i] << " to: "
                                    << std::setprecision(8)<< m_b[L][i] + negativeBiasesInfluence[i] * LEARNING_RATE<< ".\n";
#endif // DEBUG_MODE_ENABLED

                        m_b[L] += negativeBiasesInfluence * LEARNING_RATE;
                }
        }

        void encode(
                const char *path
        ) {
                constexpr uint32_t ENCODE_INFO[3] {LAYER_COUNT, WEIGHTS_COUNT, BIASES_COUNT };
#ifdef DEBUG_MODE_ENABLED
                Log << Logger::pref() << "Encoding current values...\n";
#endif // DEBUG_MODE_ENABLED

                // The file size is equal to the sum of the arrays and their sizes.
                MultiUseBuffer<
                        3 * sizeof(uint32_t),
                        LAYER_COUNT * sizeof(uint32_t),
                        WEIGHTS_COUNT * sizeof(float),
                        BIASES_COUNT * sizeof(float)
                > buffer{};

                std::memcpy(buffer.template get<0>(), ENCODE_INFO, 3 * sizeof(uint32_t));
                std::memcpy(buffer.template get<1>(), s_n, LAYER_COUNT * sizeof(int32_t));

                // There is no quick way to copy all matrices and vectors to a single buffer
                // so we iterate through all of them. L = 0 will give an empty matrix and
                // vector but it looks better if it starts from 0.
                for (uint32_t wOffset { 0 }, bOffset { 0 }, L { 0 }; L < LAYER_COUNT; L++) {
                        std::memcpy((float *)buffer.template get<2>() + wOffset, m_w[L].data(), m_w[L].size() * sizeof(float));
                        wOffset += m_w[L].size();

                        std::memcpy((float *)buffer.template get<3>() + bOffset, m_b[L].data(), m_b[L].size() * sizeof(float));
                        bOffset += m_b[L].size();
                }

                std::ofstream outFile(path, std::ios::binary);
                outFile.write((char *)buffer.template get<0>(), (int64_t)buffer.size);
                outFile.close();
#ifdef DEBUG_MODE_ENABLED
                Log << Logger::pref() << "Current values encoded.\n";
#endif // DEBUG_MODE_ENABLED
        }
};
