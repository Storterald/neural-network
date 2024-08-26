#include "Train.h"

void Train::supervisedTraining(
        Network &network,
        uint32_t sampleCount,
        const float *inputs,
        const float *outputs
) {
#ifdef DEBUG_MODE_ENABLED
        // Since DEBUG mode uses the Logger, the program must
        // run in single thread to avoid gibberish in the log file.
        constexpr uint32_t THREAD_COUNT { 1 };
#else
        // If there are less samples than threads allocate
        // sampleCount threads.
        const uint32_t THREAD_COUNT { std::min(std::thread::hardware_concurrency(), sampleCount) };
#endif // DEBUG_MODE_ENABLED

        const uint32_t BATCH_SIZE { sampleCount / THREAD_COUNT };
        const uint32_t REMAINDER { sampleCount % THREAD_COUNT };

        std::vector<std::future<void>> futures;
        futures.reserve(THREAD_COUNT);

        // Process batches in parallel using all available threads
        for (uint32_t t { 0 }; t < THREAD_COUNT; t++) {
                const uint32_t START { t * BATCH_SIZE };
                const uint32_t END { START + BATCH_SIZE + (t < REMAINDER ? 1 : 0) };

                futures.emplace_back(std::async(std::launch::async, [&]() -> void {
                        for (uint32_t i { START }; i < END; i++)
                                network._backpropagate(inputs + i * network.m_inputSize, outputs + i * network.m_outputSize);
                }));
        }

        // Wait for all threads to finish
        for (std::future<void> &f : futures)
                f.get();
}

void Train::PPOTraining(
        Network &policyNetwork,
        Network &valueNetwork,
        uint32_t sampleCount,
        const float *inputs
) {

}