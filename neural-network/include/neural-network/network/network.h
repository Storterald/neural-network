#pragma once

#include <string_view>
#include <filesystem>
#include <utility>  // std::forward
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

class network {
public:
        template<std::ranges::range layers_t>
        requires std::is_same_v<typename layers_t::value_type, layer_create_info>
        network(
                const layers_t                     &layers,
                const std::filesystem::path        &path = "") :
                network(std::ranges::begin(layers), std::ranges::end(layers), path) {}

        template<typename iterator_t>
        requires std::is_same_v<typename iterator_t::value_type, layer_create_info>
        network(
                const iterator_t                   &begin,
                const iterator_t                   &end,
                const std::filesystem::path        &path = "") :
                network(std::to_address(begin), std::to_address(end), path) {}

        network(
                const layer_create_info            *begin,
                const layer_create_info            *end,
                const std::filesystem::path        &path = "");

        [[nodiscard]] vector forward(vector input) const;
        void backward(const vector &input, const vector &cost);
        void backward(vector cost, const vector activations[]);

        void train_supervised(
                uint32_t           samples,
                const float        inputs[],
                const float        outputs[]);

        template<std::derived_from<environment> env_t, typename ...Args>
        inline void train_ppo(
                network         &value,
                uint32_t        epochs,
                uint32_t        max_steps,
                Args            &&...args) {

                env_t env(std::forward<Args...>(args)...);
                _train_ppo(value, env, epochs, max_steps);
        }

        void encode(const std::filesystem::path &path) const;

        [[nodiscard]] inline uint32_t layer_count() const
        {
                return m_count;
        }

        [[nodiscard]] inline uint32_t input_size() const
        {
                return m_L[0]->size();
        }

        [[nodiscard]] inline uint32_t output_size() const
        {
                return m_L[m_count - 1]->size();
        }

        ~network();

private:
        using chunk = std::ranges::subrange<std::span<const float>::iterator>;

        uint32_t               m_count;
        stream                 m_stream;
        std::unique_ptr<layer> *m_L;

        [[nodiscard]] stream _create_stream(const layer_create_info *infos) const;

        [[nodiscard]] std::unique_ptr<layer> *_create_layers(
                const layer_create_info        *infos) const;

        [[nodiscard]] std::unique_ptr<layer> *_create_layers(
                const layer_create_info        *infos,
                const std::string_view         &path) const;

        [[nodiscard]] std::future<void> _get_supervised_future(
                const chunk        &inputs,
                const chunk        &outputs);

        inline void _forward_impl(
                const vector        &input,
                vector              a[]) const;

        void _train_ppo(
                network            &value,
                environment        &env,
                uint32_t           epochs,
                uint32_t           max_steps);

        void _train_ppo_iteration(
                network                                     &value,
                environment                                 &env,
                uint32_t                                    max_steps,
                std::unordered_map<uint64_t, vector>        &policies);

}; // class network

} // namespace nn
