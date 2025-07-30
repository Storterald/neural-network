#include <filesystem>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <array>
#include <span>

#include <neural-network/network/network.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>

namespace fs = std::filesystem;
namespace ch = std::chrono;

#define MNIST_TRAINING
static constexpr uint32_t MAX_ITERATIONS = 1;

#if defined(MNIST_TRAINING) && defined(PPO_TRAINING)
#error "Only one training mode can be selected at a time."
#endif // MNIST_TRAINING && PPO_TRAINING

#ifdef MNIST_TRAINING
static void _get_inputs(
        std::string_view   path,
        uint32_t           sampleCount,
        std::vector<float> &inputs,
        std::vector<float> &outputs) {

        inputs.resize((size_t)sampleCount * 784, 0);
        outputs.resize((size_t)sampleCount * 10, 0);

        const fs::path cwd      = fs::path(__FILE__).parent_path();
        const fs::path filePath = cwd / ".." / ".." / "mnist" / path;
        std::ifstream file(filePath, std::ios::binary);

        int tmp;
        for (uint32_t i = 0; i < sampleCount; ++i) {
                file.read((char *)&tmp, sizeof(int));
                outputs[i * 10 + tmp] = 1.0f;
                file.read((char *)&inputs[(size_t)i * 784], sizeof(float) * 784);
        }
}

template<typename ...Args>
static void _train(
        const fs::path &encoded,
        Args           &&...args) {

        constexpr uint32_t sampleCount = 10000;

        std::vector<float> inputs, outputs;
        _get_inputs("mnist_train.nntv", sampleCount, inputs, outputs);

        nn::network network(std::forward<Args>(args)..., fs::exists(encoded) ? encoded : "");
        const auto start = ch::high_resolution_clock::now();

        for (uint32_t i = 0; i < MAX_ITERATIONS; ++i) {
                network.train_supervised(sampleCount, inputs.data(), outputs.data());
                nn::logger::log() << nn::logger::pref(nn::info) << "Iteration " << i
                                  << " completed.\n";
        }

        const auto end = ch::high_resolution_clock::now();
        std::cout << "Training completed in " << ch::duration_cast<ch::milliseconds>(end - start)
                  << ".\n";

        network.encode(encoded);
}

template<typename ...Args>
static void _test(
        const fs::path &encoded,
        Args           &&...args) {

        constexpr uint32_t sampleCount = 1000;

        std::vector<float> inputs, outputs;
        _get_inputs("mnist_test.nntv", sampleCount, inputs, outputs);

        const nn::network network(args..., fs::exists(encoded) ? encoded : "");

        long long us = 0;

        uint32_t correct = 0;
        for (uint32_t i = 0; i < sampleCount; ++i) {
                const auto start = ch::high_resolution_clock::now();
                std::span tmp(&outputs[(size_t)i * 10], 10);
                nn::vector in(784, &inputs[(size_t)i * 784]);
                nn::vector out = network.forward(in);
                const auto end = ch::high_resolution_clock::now();

                us += ch::duration_cast<ch::microseconds>(end - start).count();

                // Not benchmarking this part as it is extremely slow on GPU and
                // in a real scenario it would be done in a kernel.
                const uint32_t output   = (uint32_t)std::distance(out.begin(), std::ranges::max_element(out));
                const uint32_t expected = (uint32_t)std::distance(tmp.begin(), std::ranges::max_element(tmp));
                correct          += output == expected;

                auto pref = output == expected ? nn::logger::pref(nn::info) : nn::logger::pref(nn::error);
                nn::logger::log() << pref << "Expected: " << expected << ", got: " << output << ", with " << out[output] << " certainty.\n";
        }

        std::cout << "Test completed in " << us << "us. Correct guesses: " << correct << "/1000.\n";
}

#endif // MNIST_TRAINING

#ifdef PPO_TRAINING
#include <neural-network/network/PPO/environment.h>

class SimpleCart final : public nn::environment {
        float              position;
        float              target;
        float              velocity;
        const float        maxVelocity = 1.0f;
        const float        tolerance = 0.01f;

public:
        SimpleCart() :
                position(0.0f),
                target(10.0f),
                velocity(0.0f) {}

        // Get the current state of the environment (just position and velocity)
        [[nodiscard]] nn::vector get_state() const override
        {
                return { position, velocity };
        }

        // Perform an action and return the reward and whether the episode is done
        std::pair<float, bool> step(const nn::vector &action) override
        {
                // Action will represent the acceleration applied to the cart
                float acceleration = action.at(0);
                velocity += acceleration;

                // Clip velocity to max limits
                velocity = std::clamp(velocity, -maxVelocity, maxVelocity);
                position += velocity;

                // Reward: Higher reward for getting closer to the target
                float reward = -std::fabs(target - position);
                bool done = target - position <= tolerance;

#ifdef DEBUG_MODE_ENABLED
                nn::logger::log() << LOGGER_PREF(DEBUG) << "Reward: " << reward << " Done: " << done << " Action: " << action << '\n';
#endif // DEBUG_MODE_ENABLED
                return { reward, done };
        }

        // Reset the environment to its initial state
        void reset() override 
        {
#ifdef DEBUG_MODE_ENABLED
                nn::logger::log() << LOGGER_PREF(DEBUG) << "IEnvironment::reset().\n";
#endif // DEBUG_MODE_ENABLED
                position = 0.0f;
                target = 10.0f;
                velocity = 0.0f;
        }
};
#endif // PPO_TRAINING

int main(int argc, const char *argv[])
{
        std::vector<std::string_view> args(argv + 1, argv + argc);

        const fs::path dir     = fs::path(__FILE__).parent_path();
        const fs::path encoded = dir / "Encoded.nnv";
        fs::create_directory(dir / "logs");

        nn::logger::log()
                .set_directory(dir / "logs")
                .set_print_on_fatal(true);

#ifdef MNIST_TRAINING
        constexpr std::array INFOS = {
                nn::layer_create_info {
                        .type  = nn::layer_type::input,
                        .count = 784
                },
                nn::layer_create_info {
                        .type  = nn::layer_type::dense,
                        .count = 16
                },
                nn::layer_create_info {
                        .type       = nn::layer_type::activation,
                        .activation = nn::function_type::tanh,
                },
                nn::layer_create_info {
                        .type  = nn::layer_type::dense,
                        .count = 10
                },
                nn::layer_create_info {
                        .type       = nn::layer_type::activation,
                        .activation = nn::function_type::relu
                }
        };

        if (std::ranges::find(args, "--train") != args.end())
                _train(encoded, INFOS);

        if (std::ranges::find(args, "--test") != args.end())
                _test(encoded, INFOS);

#endif // MNIST_TRAINING
#ifdef PPO_TRAINING
        constexpr uint32_t MAX_ITERATIONS                      = 1000;
        constexpr uint32_t LAYER_COUNT                         = 3;
        constexpr uint32_t SIZES[LAYER_COUNT]                  = { 2, 16, 1 };
        constexpr nn::layer_create_info INFOS[LAYER_COUNT - 1] = {
                { .type = nn::FULLY_CONNECTED, .functionType = nn::RELU, .neuronCount = SIZES[1] },
                { .type = nn::FULLY_CONNECTED, .functionType = nn::TANH, .neuronCount = SIZES[2] }
        };

        nn::network policyNetwork(SIZES[0], LAYER_COUNT - 1, INFOS);
        nn::network valueNetwork(SIZES[0], LAYER_COUNT - 1, INFOS);

        if constexpr (IN_TRAINING) {
                const auto start = ch::system_clock::now();

                policyNetwork.train_ppo<SimpleCart>(
                        valueNetwork, MAX_ITERATIONS, 1000);
                policyNetwork.encode(dir / "Encoded.nnv");
                valueNetwork.encode(dir / "Encoded-Value.nnv");

                const auto end = ch::system_clock::now();
                std::cout << "Training completed in " << ch::duration_cast<ch::microseconds>(end - start) << ".\n";
        } else {
                SimpleCart env{};
                for (bool done = false; !done;) {
                        const nn::vector action =  policyNetwork.forward(env.getState());
                        auto [_, _done] = env.step(action);
                        done = _done;
                }
        }
#endif // PPO_TRAINING

        return EXIT_SUCCESS;
}
