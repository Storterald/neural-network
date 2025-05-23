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
#include <neural-network/base.h>

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

        [[nodiscard]] vector forward(const vector &input) const;
        void backward(const vector &input, const vector &cost);
        void backward(const vector &cost, const vector activationValues[]);

        void train_supervised(
                uint32_t           samplesCount,
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
        using chunk = std::ranges::subrange<std::span<const float>::iterator>;

        uint32_t                       m_layerCount;

#ifdef BUILD_CUDA_SUPPORT
        stream                         m_stream;
#else
        static constexpr stream        m_stream = invalid_stream;
#endif // BUILD_CUDA_SUPPORT
        uint32_t                       m_inputSize;
        uint32_t                       m_outputSize;
        std::unique_ptr<layer>         *m_L;  // [m_layerCount - 1]
        uint32_t                       *m_n;  // [m_layerCount]

#ifdef BUILD_CUDA_SUPPORT
        [[nodiscard]] stream _create_stream(const layer_create_info *infos) const;
#endif // BUILD_CUDA_SUPPORT

        [[nodiscard]] std::unique_ptr<layer> *_create_layers(
                const layer_create_info        *infos) const;

        [[nodiscard]] std::unique_ptr<layer> *_create_layers(
                const layer_create_info        *infos,
                const std::string_view         &path) const;

        [[nodiscard]] uint32_t *_get_sizes(
                const layer_create_info        *infos) const;

        [[nodiscard]] std::future<void> _get_supervised_future(
                const chunk        &inputs,
                const chunk        &outputs);

        [[nodiscard]] vector _forward_impl(vector aL) const;
        void _backward_impl(vector dC, const vector a[]);

        void _train_ppo(
                network            &valueNetwork,
                environment        &environment,
                uint32_t           epochs,
                uint32_t           maxSteps);

}; // class network

} // namespace nn
