#pragma once

#include "neural-network/utils/exceptions.h"
#include <unordered_map>
#include <type_traits>
#include <string_view>
#include <filesystem>
#include <utility>  // std::forward
#include <cstdint>
#include <fstream>
#include <ranges>
#include <future>
#include <memory>
#include <vector>
#include <span>

#include <neural-network/network/layers/activation_layer.h>
#include <neural-network/network/layers/dense_layer.h>
#include <neural-network/network/layers/input_layer.h>
#include <neural-network/network/PPO/environment.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

#ifndef DEBUG_MODE_ENABLED
#include <thread>
#endif // !DEBUG_MODE_ENABLED

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

template<floating_type _type>
class network {
public:
        using value_type = _type;

        template<std::ranges::range layers_t>
        requires std::is_same_v<typename layers_t::value_type, layer_create_info>
        network(
                const layers_t              &layers,
                const std::filesystem::path &path = "") :
                network(std::ranges::begin(layers), std::ranges::end(layers), path) {}

        template<typename iterator_t>
        requires std::is_same_v<typename iterator_t::value_type, layer_create_info>
        network(
                const iterator_t            &begin,
                const iterator_t            &end,
                const std::filesystem::path &path = "") :
                network(std::to_address(begin), std::to_address(end), path) {}

        network(
                const layer_create_info     *begin,
                const layer_create_info     *end,
                const std::filesystem::path &path = "");

        [[nodiscard]] vector<value_type> forward(vector<value_type> input) const;
        void backward(const vector<value_type> &input, const vector<value_type> &cost);
        void backward(vector<value_type> cost, const vector<value_type> activations[]);

        void train_supervised(
                uint32_t    samples,
                const float inputs[],
                const float outputs[]);

        template<std::derived_from<environment<value_type>> env_t, typename ...args_t>
        inline void train_ppo(
                network  &value,
                uint32_t epochs,
                uint32_t max_steps,
                args_t   &&...args);

        void encode(const std::filesystem::path &path) const;

        [[nodiscard]] inline uint32_t layer_count() const;
        [[nodiscard]] inline uint32_t input_size() const;
        [[nodiscard]] inline uint32_t output_size() const;

        ~network();

private:
        using chunk = std::ranges::subrange<typename std::span<std::add_const_t<value_type>>::iterator>;

        uint32_t                           m_count;
        stream_t                           m_stream;
        std::unique_ptr<layer<value_type>> *m_L;

        [[nodiscard]] stream_t _create_stream(const layer_create_info *infos) const;

        [[nodiscard]] std::unique_ptr<layer<value_type>> *_create_layers(
                const layer_create_info *infos,
                const std::string_view  &path) const;

        [[nodiscard]] std::future<void> _get_supervised_future(
                const chunk &inputs,
                const chunk &outputs);

        inline void _forward_impl(
                const vector<value_type> &input,
                vector<value_type>       a[]) const;

        void _train_ppo(
                network                 &value,
                environment<value_type> &env,
                uint32_t                epochs,
                uint32_t                max_steps);

        void _train_ppo_iteration(
                network                                          &value,
                environment<value_type>                          &env,
                uint32_t                                         max_steps,
                std::unordered_map<uint64_t, vector<value_type>> &policies);

}; // class network

template<floating_type _type>
network<_type>::network(
        const layer_create_info     *begin,
        const layer_create_info     *end,
        const std::filesystem::path &path) :

        m_count(static_cast<uint32_t>(std::distance(begin, end))),
        m_stream(_create_stream(begin)),
        m_L(_create_layers(begin, path.string())) {}

template<floating_type _type>
network<_type>::~network()
{
        delete[] m_L;
}

template<floating_type _type>
vector<_type> network<_type>::forward(vector<_type> aL) const
{
        for (uint32_t L = 1; L < m_count; ++L)
                aL = m_L[L]->forward(aL);

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT

        return aL;
}

template<floating_type _type>
void network<_type>::backward(const vector<_type> &a0, const vector<_type> &dC)
{
        const auto a = new vector<_type>[m_count];
        _forward_impl(a0, a);

        this->backward(dC, a);
        delete [] a;
}

template<floating_type _type>
void network<_type>::backward(vector<_type> dC, const vector<_type> a[])
{
        for (int32_t L = (int32_t)m_count - 1; L > 0; --L)
                dC = m_L[L]->backward(dC, a[L]);

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT
}

template<floating_type _type>
void network<_type>::train_supervised(
        uint32_t    samples,
        const float inputs[],
        const float outputs[]) {

        namespace ch = std::chrono;

#ifdef DEBUG_MODE_ENABLED
        constexpr uint32_t chunks = 1;
#else
        const uint32_t threads = std::thread::hardware_concurrency();
        const uint32_t chunks  = std::clamp(threads, 1u, samples);
#endif // DEBUG_MODE_ENABLED

        std::vector<std::future<void>> futures;
        futures.reserve(chunks);

        std::span inview(inputs,   (size_t)samples * this->input_size());
        std::span outview(outputs, (size_t)samples * this->output_size());

        const auto in  = inview  | std::views::chunk(static_cast<ptrdiff_t>(inview.size()) / chunks);
        const auto out = outview | std::views::chunk(static_cast<ptrdiff_t>(outview.size()) / chunks);
        for (uint32_t t = 0; t < chunks; ++t)
                futures.push_back(_get_supervised_future(in[t], out[t]));

        for (std::future<void> &f : futures)
                f.get();

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT
}

template<floating_type _type>
template<std::derived_from<environment<_type>> env_t, typename ...args_t>
inline void network<_type>::train_ppo(
        network  &value,
        uint32_t epochs,
        uint32_t max_steps,
        args_t   &&...args) {

        env_t env(std::forward<args_t...>(args)...);
        _train_ppo(value, env, epochs, max_steps);
}

template<floating_type _type>
void network<_type>::encode(const std::filesystem::path &path) const
{
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw fatal_error("Error opening file.");

        for (uint32_t L = 0; L < m_count; ++L)
                m_L[L]->encode(file);

        file.close();
}

template<floating_type _type>
inline uint32_t network<_type>::layer_count() const
{
        return m_count;
}

template<floating_type _type>
inline uint32_t network<_type>::input_size() const
{
        return m_L[0]->size();
}

template<floating_type _type>
inline uint32_t network<_type>::output_size() const
{
        return m_L[m_count - 1]->size();
}

template<floating_type _type>
stream_t network<_type>::_create_stream([[maybe_unused]] const layer_create_info *infos) const
{
#ifdef BUILD_CUDA_SUPPORT
        static std::array<stream_t, 16> streams = [] {
                std::array<stream_t, 16> ret{};
                for (stream_t &str : ret)
                        str = cuda::create_stream();

                return ret;
        }();
        static uint32_t usedStreams = 0;

        uint32_t ops = 0;
        for (uint32_t i = 1; i < m_count; ++i)
                ops += infos[i - 1].count * infos[i].count;

        if (ops >= CUDA_MINIMUM) {
                const stream_t str = streams[usedStreams];
                usedStreams        = (usedStreams + 1) % streams.size();
                return str;
        }

#endif // BUILD_CUDA_SUPPORT
        return invalid_stream;
}

template<floating_type _type>
std::unique_ptr<layer<_type>> *network<_type>::_create_layers(
        const layer_create_info *infos,
        const std::string_view  &path) const {

        namespace fs = std::filesystem;

        if (infos[0].type != layer_type::input)
                throw fatal_error("First network layer must be a input one.");

        const auto layers = new std::unique_ptr<layer<_type>>[m_count]();
        layers[0]         = std::make_unique<input_layer<_type>>(infos[0].count);
        uint32_t prev     = infos[0].count;

        std::ifstream istream = {};
        if (!path.empty()) {
                if (!fs::exists(path))
                        throw fatal_error("File given to network() does not exist.");
                istream = std::ifstream(fs::path(path), std::ios::binary);
        }

        for (uint32_t i = 1; i < m_count; ++i) {
                switch (infos[i].type) {
                case input:
                        throw fatal_error("Only the first layer can be an input one.");
                case dense:
                        if (istream)
                                layers[i] = std::make_unique<dense_layer<_type>>(prev, infos[0].count, istream, m_stream);
                        else
                                layers[i] = std::make_unique<dense_layer<_type>>(prev, infos[0].count, m_stream);
                        prev      = infos[0].count;
                        break;
                case activation:
                        layers[i] = std::make_unique<activation_layer<_type>>(infos[0].activation);
                        break;
                }
        }

        return layers;
}

template<floating_type _type>
std::future<void> network<_type>::_get_supervised_future(
        const chunk &inputs,
        const chunk &outputs) {

        return std::async(std::launch::async, [&]() -> void {
                const size_t size = inputs.size() / this->input_size();

                for (uint32_t i = 0; i < size; ++i) {
                        const float *in  = inputs.data() + (uintptr_t)i * this->input_size();
                        const float *out = outputs.data() + (uintptr_t)i * this->output_size();

                        const auto a = new vector<_type>[m_count];
                        _forward_impl(vector<_type>(in, in + this->input_size(), m_stream), a);

                        const vector<_type> y(out, out + this->output_size(), m_stream);
                        const vector dC = 2.0f * (a[m_count - 1] - y);

                        this->backward(dC, a);
                        delete [] a;
                }
        });
}

template<floating_type _type>
inline void network<_type>::_forward_impl(
        const vector<_type> &input,
        vector<_type>       a[]) const {

        a[0] = input;
        for (uint32_t L = 1; L < m_count; ++L)
                a[L] = m_L[L - 1]->forward(a[L - 1]);
}

template<floating_type _type>
struct _iteration_data {
        uint64_t      stateHash;
        float         reward;
        float         predicted;
        vector<_type> policy;
        vector<_type> *pa;
        vector<_type> *va;

};

template<floating_type _type>
void network<_type>::_train_ppo(
        network            &value,
        environment<_type> &env,
        uint32_t           epochs,
        uint32_t           max_steps) {

        std::unordered_map<uint64_t, vector<_type>> policies;
        for (uint32_t i = 0; i < epochs; ++i)
                _train_ppo_iteration(value, env, max_steps, policies);
}

template<floating_type _type>
void network<_type>::_train_ppo_iteration(
        network                                     &value,
        environment<_type>                          &env,
        uint32_t                                    max_steps,
        std::unordered_map<uint64_t, vector<_type>> &policies) {

        std::vector<_iteration_data<_type>> iteration;
        for (uint32_t s = 0, done = false; !done && s < max_steps; ++s) {
                iteration.emplace_back();
                auto &[hash, reward, predicted, policy, pa, va] = iteration.back();

                const vector state = env.get_state();
                hash               = std::hash<buf<_type>>{}(state);

                pa = new vector<_type>[m_count];
                _forward_impl(state, pa);

                va = new vector<_type>[value.m_count];
                value._forward_impl(state, va);

                policy    = pa[m_count - 1];
                predicted = va[value.m_count - 1].at(0);

                auto [_reward, _done] = env.step(pa[m_count - 1]);
                reward                = _reward;
                done                  = _done;
        }

        const auto states = (uint32_t)iteration.size();
        float *advantages = new float[states], previous = 0.0f;

        for (uint32_t s = states - 1; s != UINT32_MAX; --s) {
                const float reward      = iteration[s].reward;
                const float predicted   = iteration[s].predicted;
                const float nextpredict = s == states - 1 ?
                        0 : iteration[s + 1].predicted;

                const float delta = reward + GAMMA * nextpredict * predicted;
                advantages[s]     = delta + GAMMA * LAMBDA * previous;
                previous          = advantages[s];
        }

        for (uint32_t s = 0; s < states; ++s) {
                auto &[hash, reward, predicted, policy, pa, va] = iteration[s];

                const float advantage = advantages[s];
                policy                = policy.max(EPSILON); // prevent division by 0.
                const vector vdC      = { (predicted - reward) * 2.0f };

                value.backward(vdC, va);
                delete [] va;

                policies.insert({hash, policy});

                const vector ratio   = policy / policies.at(hash);
                const vector clipped = ratio.clamp(1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON);
                const vector loss    = (ratio * advantage).min(clipped * advantage);

                policies[hash] = policy;

                this->backward(loss, pa);
                delete [] pa;
        }

        delete [] advantages;
        env.reset();
}

} // namespace nn
