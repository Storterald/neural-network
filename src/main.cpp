#include <iostream>

#include "network/Train.h"
#include "utils/Logger.h"

#define __BENCHMARK_START(__NAME__) auto __NAME__ = std::chrono::high_resolution_clock::now()
#define __BENCHMARK_END(__START_NAME__, __OUT_NAME__) std::chrono::duration<double> __OUT_NAME__ = std::chrono::high_resolution_clock::now() - __START_NAME__;

constexpr uint32_t LAYER_COUNT { 4 };
constexpr uint32_t SIZES[LAYER_COUNT] { 784, 16, 16, 10 };

void logComputeInfo(
        const float *inputs,
        const float *outputs,
        const Vector &result,
        uint32_t i
) {
        constexpr uint32_t INPUT_NEURONS { SIZES[0] };
        constexpr uint32_t OUTPUT_NEURONS { SIZES[LAYER_COUNT - 1] };

        Log << Logger::pref<INFO>() << "Iteration [" << i << "]...\n";

        Log << Logger::pref() << "Inputs: ";
        const float *const ptr { inputs + i * INPUT_NEURONS };
        for (uint32_t j { 0 }; j < INPUT_NEURONS; j++)
                Log << ptr[j] << ", ";
        Log << "\n";

        Log << Logger::pref<INFO>() << "Output: ";
        for (uint32_t j { 0 }; j < OUTPUT_NEURONS; j++)
                Log << result.at(j) << ", ";
        Log << "\n";

        const float *const expectedOutput { outputs + i * OUTPUT_NEURONS };
        const uint32_t index { (uint32_t)std::distance(
                expectedOutput, std::find(expectedOutput, expectedOutput + OUTPUT_NEURONS, 1.0f)
        )};
        Log << Logger::pref<INFO>() << "Correct answer: " << index << ", confidence: " << result.at(index) << ", best option: ";

        // Implementing max like this since std::max_element does not work
        // on GPU memory, this way it works also when compiling with USE_CUDA
        float highest { result.at(index) };
        for (uint32_t j { 0 }; j < OUTPUT_NEURONS; j++)
                if (const float v { result.at(j) }; v > highest)
                        highest = v;

        Log << (result.at(index) == highest && result.at(index) != 0.0f ? "true" : "false") << "\n";
        Log << Logger::pref<INFO>() << "\n";
}

int main()
{
        constexpr bool IN_TRAINING { false };
        constexpr uint32_t MAX_ITERATIONS { 1000 };

        constexpr uint32_t SAMPLE_COUNT { IN_TRAINING ? 60000 : 10000 };
        constexpr uint32_t INPUT_NEURONS { SIZES[0] };
        constexpr uint32_t OUTPUT_NEURONS { SIZES[LAYER_COUNT - 1] };

        std::ifstream inFile(IN_TRAINING ? BASE_PATH "/mnist/mnist_train.nntv" : BASE_PATH "/mnist/mnist_test.nntv", std::ios::binary);
        if (!inFile)
                return EXIT_FAILURE;

        std::vector inputs(SAMPLE_COUNT * INPUT_NEURONS, 0.0f), outputs(SAMPLE_COUNT * OUTPUT_NEURONS, 0.0f);
        for (uint32_t i { 0 }; i < SAMPLE_COUNT; i++) {
                uint32_t correctOutput{};
                inFile.read((char *)&correctOutput, sizeof(uint32_t));
                float *const outputPtr { outputs.data() + i * OUTPUT_NEURONS };
                outputPtr[correctOutput] = 1.0f;

                float *const inputPtr { inputs.data() + i * INPUT_NEURONS };
                inFile.read((char *)inputPtr, INPUT_NEURONS * sizeof(float));
        }
        inFile.close();

        LayerCreateInfo infos[LAYER_COUNT - 1] {
                { FULLY_CONNECTED, TANH, SIZES[1] },
                { FULLY_CONNECTED, TANH, SIZES[2] },
                { FULLY_CONNECTED, TANH, SIZES[3] }
        };

        for (uint32_t i { 0 }; i < (IN_TRAINING ? MAX_ITERATIONS : 1); i++) {
                Network network(SIZES[0], LAYER_COUNT - 1, infos, BASE_PATH "/Encoded.nnv");

                __BENCHMARK_START(start);

                if constexpr (IN_TRAINING) {
                        Train::supervisedTraining(network, SAMPLE_COUNT, inputs.data(), outputs.data());
                } else for (uint32_t j { 0 }; j < SAMPLE_COUNT; j++) {
                        const Vector result { network.compute(inputs.data() + j * INPUT_NEURONS) };
                        logComputeInfo(inputs.data(), outputs.data(), result, j);
                }

                __BENCHMARK_END(start, time)

                if constexpr (IN_TRAINING)
                        network.encode(BASE_PATH "/Encoded.nnv");

                std::cout << "Iteration [" << i << "] Computed In " << Logger::fixTime(time.count()) << "\n" << std::flush;
        }

        return EXIT_SUCCESS;
}