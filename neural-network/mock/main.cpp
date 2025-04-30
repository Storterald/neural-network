#include <neural-network/network/PPO/environment.h>
#include <neural-network/network/network.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>

#include <filesystem>
#include <iostream>
#include <chrono>

namespace fs = std::filesystem;
namespace ch = std::chrono;

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

int main()
{
        const fs::path dir = fs::path(__FILE__).parent_path();
        fs::create_directory(dir / "logs");
        nn::logger::log().set_directory(dir / "logs");

        constexpr bool IN_TRAINING                             = true;
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
}
