#include "network/Train.h"

#ifdef DEBUG_MODE_ENABLED
#include "utils/Logger.h"
#endif // DEBUG_MODE_ENABLED

class SimpleCart : public IEnvironment {
        float              position;
        float              target;
        float              velocity;
        const float        maxVelocity = 1.0f;
        const float        tolerance = 0.01f;

public:
        SimpleCart() :
                position(0.0f),
                target(10.0f),
                velocity(0.0f)
        {}

        // Get the current state of the environment (just position and velocity)
        [[nodiscard]] Vector getState() const override 
        {
                return { position, velocity };
        }

        // Perform an action and return the reward and whether the episode is done
        std::pair<float, bool> step(const Vector &action) override 
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
                Log << LOGGER_PREF(DEBUG) << "Reward: " << reward << " Done: " << done << " Action: " << action << '\n';
#endif // DEBUG_MODE_ENABLED
                return { reward, done };
        }

        // Reset the environment to its initial state
        void reset() override 
        {
#ifdef DEBUG_MODE_ENABLED
                Log << LOGGER_PREF(DEBUG) << "IEnvironment::reset().\n";
#endif // DEBUG_MODE_ENABLED
                position = 0.0f;
                target = 10.0f;
                velocity = 0.0f;
        }
};

int main()
{
        constexpr bool IN_TRAINING { true };
        constexpr uint32_t MAX_ITERATIONS { 1000 };
        constexpr uint32_t LAYER_COUNT { 3 };
        constexpr uint32_t SIZES[LAYER_COUNT] { 2, 16, 1 };
        constexpr LayerCreateInfo INFOS[LAYER_COUNT - 1] {
                { .type = FULLY_CONNECTED, .functionType = RELU, .neuronCount = SIZES[1] },
                { .type = FULLY_CONNECTED, .functionType = TANH, .neuronCount = SIZES[2] }
        };

        Network policyNetwork(
                SIZES[0], LAYER_COUNT - 1,
                INFOS, BASE_PATH "/Encoded.nnv");
        Network valueNetwork(
                SIZES[0], LAYER_COUNT - 1,
                INFOS, BASE_PATH "/Encoded-Value.nnv");

        if constexpr (IN_TRAINING) {
                Train::PPO<SimpleCart>(policyNetwork, valueNetwork, MAX_ITERATIONS, 1000);
                policyNetwork.encode(BASE_PATH "/Encoded.nnv");
                valueNetwork.encode(BASE_PATH "/Encoded-Value.nnv");
        } else {
                SimpleCart env{};
                for (bool done { false }; !done;) {
                        const Vector action { policyNetwork.forward(env.getState()) };
                        auto [_, _done] { env.step(action) };
                        done = _done;
                }
        }

        return EXIT_SUCCESS;
}