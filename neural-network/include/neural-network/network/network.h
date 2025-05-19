#pragma once

#include <string_view>
#include <filesystem>
#include <cstdint>
#include <ranges>
#include <future>
#include <memory>
#include <span>

#include <neural-network/network/PPO/environment.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>

namespace nn {

template<typename T>
concept network_infos = std::ranges::range<T>
        && std::same_as<std::iter_value_t<T>, layer_create_info>
        && !std::same_as<T, std::span<layer_create_info>>;

class network {
public:
        network() = default;

        inline network(
                uint32_t                           inputSize,
                const network_infos auto           &infos,
                const std::filesystem::path        &path = "") :
                network(inputSize, std::to_address(std::ranges::begin(infos)), std::to_address(std::ranges::end(infos)), path) {}

        network(
                uint32_t                           inputSize,
                const layer_create_info            *infosBegin,
                const layer_create_info            *infosEnd,
                const std::filesystem::path        &path = "");

        [[nodiscard]] vector forward(vector input) const;
        void backward(const vector &input, const vector &cost);
        void backward(vector cost, const vector activationValues[]);

        void train_supervised(
                uint32_t           sampleCount,
                const float        inputs[],
                const float        outputs[]);

        template<std::derived_from<environment> Environment, typename ...Ts>
        inline void train_ppo(
                network         &valueNetwork,
                uint32_t        epochs,
                uint32_t        maxSteps,
                Ts              ...args) {

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

        ~network();

private:
        uint32_t                      m_layerCount = 0;
        uint32_t                      m_inputSize  = 0;
        uint32_t                      m_outputSize = 0;
        std::unique_ptr<layer>        *m_L         = nullptr;  // [m_layerCount - 1]
        uint32_t                      *m_n         = nullptr;  // [m_layerCount]

        [[nodiscard]] std::unique_ptr<layer> *_create_layers(
                const layer_create_info        *infos) const;

        [[nodiscard]] std::unique_ptr<layer> *_create_layers(
                const layer_create_info        *infos,
                const std::string_view         &path) const;

        [[nodiscard]] uint32_t *_get_sizes(
                const layer_create_info        *infos) const;

        [[nodiscard]] std::future<void> _get_supervised_future(
                uint32_t           start,
                uint32_t           end,
                const float        inputs[],
                const float        outputs[]);

        void _train_ppo(
                network            &valueNetwork,
                environment        &environment,
                uint32_t           epochs,
                uint32_t           maxSteps);

}; // class network

} // namespace nn
