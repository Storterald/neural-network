#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>

#include "network/Network.h"
#include "utils/Logger.h"

#define __BENCHMARK_START(__NAME__) auto __NAME__ = std::chrono::high_resolution_clock::now()
#define __BENCHMARK_END(__START_NAME__, __OUT_NAME__) std::chrono::duration<double> __OUT_NAME__ = std::chrono::high_resolution_clock::now() - __START_NAME__;

constexpr uint32_t LAYER_COUNT { 4 };
constexpr uint32_t SIZES[LAYER_COUNT] { 784, 32, 32, 10 };

std::vector<float> weights([&]() -> uint32_t {
        uint32_t count { 0 };
        for (uint32_t i { 0 }; i < LAYER_COUNT - 1; i++)
                count += SIZES[i] * SIZES[i+1];

        return count;
}()), biases([&]() -> uint32_t {
        uint32_t count { 0 };
        for (uint32_t i { 1 }; i < LAYER_COUNT; i++)
                count += SIZES[i];

        return count;
}());

template<bool training>
bool load(
        uint32_t SAMPLE_COUNT,
        uint32_t INPUT_NEURONS,
        float *inputs,
        uint32_t OUTPUT_NEURONS,
        float *outputs
) {
        constexpr const char *PATH { training ? BASE_PATH "/mnist/mnist_train.nntv" : BASE_PATH "/mnist/mnist_test.nntv" };

        std::ifstream inFile(PATH, std::ios::binary);
        if (!inFile)
                return false;

        for (uint32_t i { 0 }; i < SAMPLE_COUNT; i++) {
                uint32_t correctOutput{};
                inFile.read((char *)&correctOutput, sizeof(uint32_t));
                float *const outputPtr { outputs + i * OUTPUT_NEURONS };
                outputPtr[correctOutput] = 1.0f;

                float *const inputPtr { inputs + i * INPUT_NEURONS };
                inFile.read((char *)inputPtr, INPUT_NEURONS * sizeof(float));
        }
        inFile.close();

        return true;
}

void logComputeInfo(
        uint32_t INPUT_NEURONS,
        uint32_t OUTPUT_NEURONS,
        const float *inputs,
        const float *outputs,
        const Vector &result,
        uint32_t i
) {
        Log << Logger::pref<INFO>() << "Iteration [" << i << "]...\n";

        // Copying output from array to vector for faster math.
        const float *const expectedOutput { outputs + i * OUTPUT_NEURONS };
#ifdef DEBUG_MODE_ENABLED
        Log << Logger::pref() << "Inputs [" << i << "]: ";
        const float *const ptr { inputs + i * INPUT_NEURONS };
        for (uint32_t j { 0 }; j < INPUT_NEURONS; j++)
                Log << ptr[j] << ", ";
        Log << "\n";
#endif

        Log << Logger::pref<INFO>() << "Output for iteration [" << i << "]: ";
        for (uint32_t j { 0 }; j < OUTPUT_NEURONS; j++)
                Log << result[j] << ", ";
        Log << "\n";

        const uint32_t index { (uint32_t)std::distance(
                expectedOutput, std::find(expectedOutput, expectedOutput + OUTPUT_NEURONS, 1.0f)
        )};
        Log << Logger::pref<INFO>() << "Correct answer: " << index << ", confidence: " << result[index] << ", best option: "
            << (result[index] == *std::max_element(result.data(), result.data() + OUTPUT_NEURONS) ? "true" : "false") << "\n";
}

int main()
{
        constexpr bool IN_TRAINING { false };
        constexpr uint32_t MAX_ITERATIONS { 400 };

        constexpr uint32_t SAMPLE_COUNT { IN_TRAINING ? 60000 : 10000 };
        constexpr uint32_t INPUT_NEURONS { SIZES[0] };
        constexpr uint32_t OUTPUT_NEURONS { SIZES[LAYER_COUNT - 1] };

        std::vector inputs(SAMPLE_COUNT * INPUT_NEURONS, 0.0f), outputs(SAMPLE_COUNT * OUTPUT_NEURONS, 0.0f);
        if(!load<IN_TRAINING>(SAMPLE_COUNT, INPUT_NEURONS, inputs.data(), OUTPUT_NEURONS, outputs.data()))
                return EXIT_FAILURE;

        for (uint32_t i { 0 }; i < (IN_TRAINING ? MAX_ITERATIONS : 1); i++) {
                Network<SIZES[0],
                        {FULLY_CONNECTED, TANH, SIZES[1]},
                        {FULLY_CONNECTED, TANH, SIZES[2]},
                        {FULLY_CONNECTED, TANH, SIZES[3]}
                >network(BASE_PATH "/decoder/Encoded.nnv");

                __BENCHMARK_START(start);

                if constexpr (IN_TRAINING) {
                        network.train(SAMPLE_COUNT, inputs.data(), outputs.data());
                } else for (uint32_t j { 0 }; j < SAMPLE_COUNT; j++) {
                        const Vector result { network.compute(inputs.data() + j * INPUT_NEURONS) };
                        logComputeInfo(INPUT_NEURONS, OUTPUT_NEURONS, inputs.data(), outputs.data(), result, j);
                }

                __BENCHMARK_END(start, time)

                if constexpr (IN_TRAINING)
                        network.encode(BASE_PATH "/decoder/Encoded.nnv");

                std::cout << "Iteration [" << i << "] Computed In " << Logger::fixTime(time.count()) << "\n" << std::flush;
        }

        return EXIT_SUCCESS;
}