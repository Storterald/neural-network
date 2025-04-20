#pragma once

#include <filesystem>
#include <future>
#include <memory>

#include "PPO/IEnvironment.h"
#include "ILayer.h"

class Network {
public:
        Network() = default;

        Network(
                uint32_t inputSize,
                uint32_t layerCount,
                const LayerCreateInfo layerInfos[],
                const std::filesystem::path &path = "");

        [[nodiscard]] Vector forward(Vector input) const;
        void backward(const Vector &input, const Vector &cost);
        void backward(Vector cost, const Vector activationValues[]);

        void train_supervised(
                uint32_t sampleCount,
                const float inputs[],
                const float outputs[]);

        template<std::derived_from<IEnvironment> Environment, typename ...Ts>
        inline void train_ppo(
                Network &valueNetwork,
                uint32_t epochs,
                uint32_t maxSteps,
                Ts ...args) {

                Environment env(args...);
                _train_ppo(valueNetwork, env, epochs, maxSteps);
        }

        void encode(const std::filesystem::path &path) const;

        [[nodiscard]] inline uint32_t layer_count() const
        {
                return m_layerCount;
        }

        [[nodiscard]] inline uint32_t input_size() const
        {
                return m_inputSize;
        }

        [[nodiscard]] inline uint32_t output_size() const
        {
                return m_outputSize;
        }

        ~Network();

private:
        uint32_t                       m_layerCount = 0;
        uint32_t                       m_inputSize  = 0;
        uint32_t                       m_outputSize = 0;
        std::unique_ptr<ILayer>        *m_L         = nullptr;  // [m_layerCount - 1]
        uint32_t                       *m_n         = nullptr;  // [m_layerCount]

        std::unique_ptr<ILayer> *_create_layers(
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos) const;

        std::unique_ptr<ILayer> *_create_layers(
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos,
                const char *path) const;

        uint32_t *_get_sizes(
                uint32_t layerCount,
                const LayerCreateInfo *layerInfos) const;

        std::future<void> _get_supervised_future(
                uint32_t start,
                uint32_t end,
                const float inputs[],
                const float outputs[]);

        void _train_ppo(
                Network &valueNetwork,
                IEnvironment &environment,
                uint32_t epochs,
                uint32_t maxSteps);

}; // class Layer