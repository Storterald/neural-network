#pragma once

#include <fstream>
#include <future>
#include <vector>
#include <array>

#include "Layer.h"
#include "../utils/Logger.h"
#include "../utils/MultiUseBuffer.h"

// While it would have been nice to have a completely dynamic
// network, it's way easier to have the layer infos as a variadic
// argument.
template<uint32_t inputSize, LayerCreateInfo ...layerInfos>
class Network {
        static_assert(sizeof...(layerInfos) > 0, "\n\n"
                "The layer count must be at least 2, the input and the output.");
public:
        Network(const char *path) :
                m_L(_createLayersPath(path))
        {}

        Vector compute(
                const float *inputs
        ) {
                Vector a[LAYER_COUNT] { Vector(inputSize, inputs) };

                // Cycle trough all layers 'L', reading layer 'L' and 'L-1' going forward
                for (uint32_t L { 0 }; L < LAYER_COUNT - 1; L++)
                        a[L + 1] = m_L[L]->forward(a[L]);

                return a[LAYER_COUNT - 1];
        }

        void train(
                uint32_t sampleCount,
                const float *inputs,
                const float *outputs
        ) const {
                constexpr uint32_t OUTPUT_NEURONS { s_n[LAYER_COUNT - 1] };

                // If there are less samples than threads allocate
                // sampleCount threads.
#ifdef DEBUG_MODE_ENABLED
                constexpr uint32_t THREAD_COUNT { 1 };
#else
                const uint32_t THREAD_COUNT { std::min(std::thread::hardware_concurrency(), sampleCount) };
#endif // DEBUG_MODE_ENABLED

                const uint32_t BATCH_SIZE { sampleCount / THREAD_COUNT };
                const uint32_t REMAINDER { sampleCount % THREAD_COUNT };

                std::vector<std::future<void>> futures;
                futures.reserve(THREAD_COUNT);

                // Process batches in parallel
                for (uint32_t t { 0 }; t < THREAD_COUNT; t++) {
                        const uint32_t START { t * BATCH_SIZE };
                        const uint32_t END { START + BATCH_SIZE + (t < REMAINDER ? 1 : 0) };

                        futures.emplace_back(std::async(std::launch::async, [&] {
                                for (uint32_t i { START }; i < END; i++)
                                        _backpropagate(inputs + i * inputSize, outputs + i * OUTPUT_NEURONS);
                        }));
                }

                // Wait for all threads to finish
                for (auto &f : futures)
                        f.get();
        }

        void encode(
                const char *path
        ) const {
                // The file must be open in binary mode, and all
                // encode function must write binary.
                std::ofstream file(path, std::ios::binary);
                if (!file)
                        throw Logger::fatal_error("Error opening file.");

                // Calls the encode function on all layers, the encode
                // function uses std::ofstream::write, writing binary and
                // moving the position of std::ofstream::tellp.
                for (const std::unique_ptr<ILayer> &layer : m_L)
                        layer->encode(file);

                file.close();
        }

private:
        static constexpr uint32_t LAYER_COUNT { sizeof...(layerInfos) + 1 };
        static constexpr uint32_t s_n[] { inputSize, layerInfos.neuronCount ... };

        const std::array<std::unique_ptr<ILayer>, LAYER_COUNT - 1> m_L{};

        template<uint32_t ...Is>
        std::array<std::unique_ptr<ILayer>, LAYER_COUNT - 1> _createLayersNew(
                std::index_sequence<Is...>
        ) {
                constexpr LayerCreateInfo infos[] { layerInfos... };
                return { std::make_unique<Layer<infos[Is].type, infos[Is].functionType>>(
                    (Is == 0 ? inputSize : infos[Is - 1].neuronCount), infos[Is].neuronCount
                )... };
        }

        std::array<std::unique_ptr<ILayer>, LAYER_COUNT - 1> _createLayersPath(
                const char *path
        ) {
                std::ifstream file(path, std::ios::binary);
                if (!file)
                        return _createLayersNew(std::make_index_sequence<LAYER_COUNT - 1>());

                return { std::make_unique<Layer<layerInfos.type, layerInfos.functionType>>(file) ... };
        }

        void _backpropagate(
                const float *inputs,
                const float *y
        ) const {
                Vector a[LAYER_COUNT] { Vector(inputSize, inputs) }, costs[LAYER_COUNT - 1]{};

                for (uint32_t L { 1 }; L < LAYER_COUNT; L++)
                        a[L] = m_L[L - 1]->forward(a[L - 1]);

                // The cost of the last layer neurons is calculated with (ajL - yj) ^ 2,
                // this mean that the derivative is equal to 2 * (ajL - y)
#ifdef USE_CUDA
                constexpr uint32_t OUTPUT_NEURONS { s_n[LAYER_COUNT - 1] };
                costs[LAYER_COUNT - 2] = (a[LAYER_COUNT - 1] - Vector(OUTPUT_NEURONS, y)) * 2.0f;
#else
                costs[LAYER_COUNT - 2] = (a[LAYER_COUNT - 1] - y) * 2.0f;
#endif // USE_CUDA

#ifdef DEBUG_MODE_ENABLED
                for (uint32_t L { LAYER_COUNT - 1 }, j { 0 }; j < OUTPUT_NEURONS; j++)
                        Log << Logger::pref() << "Cost for output neuron [" << j << "] has value of: " << std::pow(a[L].at(j) - y[j], 2.0f)
                            << ", and it's derivative has value of: " << costs[LAYER_COUNT - 2].at(j) << ". Formulas: (" << a[L].at(j) << " - " << y[j]
                            << ")^2, 2 * (" << a[L].at(j) << " - " << y[j] << ").\n";
#endif // DEBUG_MODE_ENABLED

                // Cycle trough all layers 'L', reading layer 'L' and 'L-1' going backward
                for (int32_t L { LAYER_COUNT - 2 }; L >= 1; L--)
                        costs[L - 1] = m_L[L]->backward(costs[L], a[L]);

                // The input layer does not have an object 'Layer'
                // so m_L[0] is the first hidden layer.
                m_L[0]->backward(costs[0], a[0]);
        }

}; // class Layer
