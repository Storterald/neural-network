#include <neural-network/network/PPO/environment.h>
#include <neural-network/network/network.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <span>

namespace fs = std::filesystem;
namespace ch = std::chrono;

#define MNIST_TRAINING

#if defined(MNIST_TRAINING) && defined(PPO_TRAINING)
#error "Only one training mode can be selected at a time."
#endif // MNIST_TRAINING && PPO_TRAINING

#ifdef MNIST_TRAINING
static void _get_inputs(
        std::string_view   path,
        uint32_t           samplesCount,
        std::vector<float> &inputs,
        std::vector<float> &outputs) {

        inputs.resize(samplesCount * 784, 0);
        outputs.resize(samplesCount * 10, 0);

        fs::path cwd = fs::path(__FILE__).parent_path();
        fs::path filePath = cwd / ".." / ".." / "mnist" / path;
        std::ifstream file(filePath, std::ios::binary);

        int tmp;
        for (uint32_t i = 0; i < samplesCount; ++i) {
                file.read((char *)&tmp, sizeof(int));
                outputs[i * 10 + tmp] = 1.0f;
                file.read((char *)&inputs[i * 784], sizeof(float) * 784);
        }
}
#endif // MNIST_TRAINING

#ifdef PPO_TRAINING
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
        [[nodiscard]] nn::vector getState() const override
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

int main()
{
        const fs::path dir = fs::path(__FILE__).parent_path();
        fs::create_directory(dir / "logs");

        nn::logger::log().set_directory(dir / "logs");
        nn::logger::log().set_print_on_fatal(true);

        constexpr bool IN_TRAINING = true;

#ifdef MNIST_TRAINING
        constexpr uint32_t MAX_ITERATIONS                      = 1000;
        constexpr uint32_t LAYER_COUNT                         = 3;
        constexpr uint32_t SIZES[LAYER_COUNT]                  = { 784, 16, 10 };
        constexpr nn::layer_create_info INFOS[LAYER_COUNT - 1] = {
                { .type = nn::FULLY_CONNECTED, .functionType = nn::TANH, .neuronCount = SIZES[1] },
                { .type = nn::FULLY_CONNECTED, .functionType = nn::RELU, .neuronCount = SIZES[2] }
        };

        std::vector<float> inputs;
        std::vector<float> outputs;

        if constexpr (IN_TRAINING) {
                nn::network network(SIZES[0], LAYER_COUNT - 1, INFOS);

                _get_inputs("mnist_train.nntv", 10000, inputs, outputs);
                const auto start = ch::system_clock::now();

                for (uint32_t i = 0; i < MAX_ITERATIONS; ++i) {
                        network.train_supervised(10000, inputs.data(), outputs.data());
                        nn::logger::log() << LOGGER_PREF(DEBUG) << "Iteration " << i << " completed.\n";
                }

                network.encode(dir / "Encoded.nnv");

                const auto end = ch::system_clock::now();
                std::cout << "Training completed in " << ch::duration_cast<ch::microseconds>(end - start) << ".\n";
        } else {
                nn::network network(SIZES[0], LAYER_COUNT - 1, INFOS, dir / "Encoded.nnv");

                _get_inputs("mnist_test.nntv", 1000, inputs, outputs);
                const auto start = ch::system_clock::now();

                for (uint32_t i = 0; i < 1000; ++i) {
                        std::span tmp(&outputs[i * 10], &outputs[i * 10 + 10]);
                        nn::vector in(784, &inputs[i * 784]);
                        nn::vector out = network.forward(in);

                        int output = (int)std::distance(out.begin(), std::ranges::max_element(out));
                        int expected = (int)std::distance(tmp.begin(), std::ranges::max_element(tmp));

                        auto pref = output == expected ? LOGGER_PREF(INFO) : LOGGER_PREF(ERROR);
                        nn::logger::log() << pref << "Expected: " << expected << ", got: " << output << ", with " << out[output] << " certainty.";
                }

                const auto end = ch::system_clock::now();
                std::cout << "Test completed in " << ch::duration_cast<ch::microseconds>(end - start) << ".\n";
        }

        return EXIT_SUCCESS;
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

        return EXIT_SUCCESS;
#endif // PPO_TRAINING
}
