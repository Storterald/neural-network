#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>

#include "Network.h"
#include "utils/Logger.h"
#include "../decoder/Values.h"

#define __BENCHMARK_START(__NAME__) auto __NAME__ = std::chrono::high_resolution_clock::now()
#define __BENCHMARK_END(__START_NAME__, __OUT_NAME__) std::chrono::duration<double> __OUT_NAME__ = std::chrono::high_resolution_clock::now() - __START_NAME__;

constexpr uint32_t LAYER_COUNT { 4 };
constexpr uint32_t SIZES[LAYER_COUNT] { 784, 16, 16, 10 };

std::vector<float> weights([] -> uint32_t {
        uint32_t count { 0 };
        for (uint32_t i { 0 }; i < LAYER_COUNT - 1; i++)
                count += SIZES[i] * SIZES[i+1];

        return count;
}()), biases([] -> uint32_t {
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
        constexpr const char *PATH { training ? (BASE_PATH "/mnist/mnist_train.nntv") : (BASE_PATH "/mnist/mnist_test.nntv") };

        std::ifstream inFile(PATH, std::ios::binary);
        if (!inFile)
                return false;

        for (uint32_t i { 0 }; i < SAMPLE_COUNT; i++) {
                uint32_t correctOutput{};
                inFile.read((char *)&correctOutput, (std::streamsize)sizeof(uint32_t));
                float *const outputPtr { outputs + i * OUTPUT_NEURONS };
                outputPtr[correctOutput] = 1.0f;

                float *const inputPtr { inputs + i * INPUT_NEURONS };
                inFile.read((char *)inputPtr, INPUT_NEURONS * (std::streamsize)sizeof(float));
        }
        inFile.close();

        return true;
}

bool loadValues()
{
        std::ifstream inFile(BASE_PATH "/decoder/Encoded.nnv", std::ios::binary);
        if (!inFile)
                return false;

        // Data sizes
        uint32_t infos[3]{};
        inFile.read((char *)&infos, 3 * sizeof(uint32_t));

        // Discard sizes array
        uint32_t *sizes { new uint32_t[infos[0]]() };
        inFile.read((char *)sizes, infos[0] * (int64_t)sizeof(uint32_t));
        delete [] sizes;

        // Read weights and biases array
        inFile.read((char *)weights.data(), infos[1] * (int64_t)sizeof(float));
        inFile.read((char *)biases.data(), infos[2] * (int64_t)sizeof(float));

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
        Vector expectedOutput(OUTPUT_NEURONS);
        std::memcpy(expectedOutput.data(), outputs + i * OUTPUT_NEURONS, OUTPUT_NEURONS * sizeof(float));

        Log << Logger::pref() << "Inputs [" << i << "]: ";
        const float *const ptr { inputs + i * INPUT_NEURONS };
        for (uint32_t j { 0 }; j < INPUT_NEURONS; j++)
                Log << ptr[j] << ", ";
        Log << "\n";

        Log << Logger::pref<INFO>() << "Output for iteration [" << i << "]: ";
        for (uint32_t j { 0 }; j < OUTPUT_NEURONS; j++)
                Log << result[j] << ", ";
        Log << "\n";

        Vector fixedResults { Fast::relu(result) };

        uint32_t index{(uint32_t) std::distance(
                expectedOutput.data(), std::find(expectedOutput.data(), expectedOutput.data() + OUTPUT_NEURONS, 1.0f)
        )};
        Log << Logger::pref<INFO>() << "Correct answer: " << index << ", confidence: " << result[index] << ", best option: "
            << (result[index] == *std::max_element(result.data(), result.data() + OUTPUT_NEURONS) ? "true" : "false") << "\n";
}

int main()
{
        constexpr bool IN_TRAINING { true };
        constexpr bool AFK_TRAINING { true };
        constexpr uint32_t MAX_ITERATIONS { 50 };

        constexpr uint32_t SAMPLE_COUNT { IN_TRAINING ? 60000 : 10000 };
        constexpr uint32_t BATCH_SIZE { 30 };
        constexpr uint32_t ITERATIONS_COUNT { IN_TRAINING ? 60000 / BATCH_SIZE : SAMPLE_COUNT };
        constexpr uint32_t INPUT_NEURONS { SIZES[0] };
        constexpr uint32_t OUTPUT_NEURONS { SIZES[LAYER_COUNT - 1] };

        std::vector<float> inputs(SAMPLE_COUNT * INPUT_NEURONS, 0.0f), outputs(SAMPLE_COUNT * OUTPUT_NEURONS, 0.0f);
        if(!load<IN_TRAINING>(SAMPLE_COUNT, INPUT_NEURONS, inputs.data(), OUTPUT_NEURONS, outputs.data()))
                return EXIT_FAILURE;

        // Initializing values with random generated / decoded ones.
        weights = Values::weights;
        biases = Values::biases;

        Network<TANH, INPUT_NEURONS, SIZES[1], SIZES[2], OUTPUT_NEURONS> network(weights.data(), biases.data());

        // Encoding values if not already encoded
        if (!std::filesystem::exists(BASE_PATH "/decoder/Encoded.nnv"))
                network.encode(BASE_PATH "/decoder/Encoded.nnv");

        for (uint32_t j { 0 }; j < (IN_TRAINING ? MAX_ITERATIONS : 1); j++) {
                if constexpr (IN_TRAINING && AFK_TRAINING)
                        if (!loadValues())
                                return EXIT_FAILURE;

                __BENCHMARK_START(start);

                for (uint32_t i { 0 }; i < ITERATIONS_COUNT; i++) {
                        if constexpr (IN_TRAINING) {
                                network.trainWithBackpropagation(BATCH_SIZE,
                                         inputs.data() + i * BATCH_SIZE * INPUT_NEURONS,
                                         outputs.data() + i * BATCH_SIZE * OUTPUT_NEURONS
                                );
                        } else {
                                Vector result { network.compute(inputs.data() + i * INPUT_NEURONS) };
                                logComputeInfo(INPUT_NEURONS, OUTPUT_NEURONS, inputs.data(), outputs.data(), result, i);
                        }
                }

                __BENCHMARK_END(start, time)

                if constexpr (IN_TRAINING)
                        network.encode(BASE_PATH "/decoder/Encoded.nnv");

                std::cout << "Computed In " << Logger::fixTime(time.count()) << "\n" << std::flush;
        }

        return EXIT_SUCCESS;
}