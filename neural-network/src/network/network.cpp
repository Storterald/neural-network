#include <neural-network/network/network.h>

#include <unordered_map>
#include <filesystem>
#include <optional>
#include <numeric> // std::accumulate
#include <fstream>
#include <cstdint>
#include <vector>
#include <thread>
#include <future>

#include <neural-network/network/PPO/environment.h>
#include <neural-network/utils/exceptions.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/utils/logger.h>
#include <neural-network/base.h>

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#include <neural-network/cuda_base.h>
#endif // BUILD_CUDA_SUPPORT

template<typename ...Args>
static void _create_layers(
        std::unique_ptr<nn::layer>         *layers,
        uint32_t                           count,
        const nn::layer_create_info        *infos,
        Args                               &&...args) {

        layers[0] = nn::layer::create(0, infos[0], std::forward<Args>(args)...);
        for (uint32_t L = 1; L < count; ++L)
                layers[L] = nn::layer::create(infos[L - 1].count, infos[L], std::forward<Args>(args)...);
}

namespace nn {

network::network(
        const layer_create_info            *begin,
        const layer_create_info            *end,
        const std::filesystem::path        &path) :

        m_count(static_cast<uint32_t>(std::distance(begin, end))),
        m_stream(_create_stream(begin)),
        m_L(_create_layers(begin, path.string()))
{}

network::~network()
{
        delete[] m_L;
}

vector network::forward(vector aL) const
{
        for (uint32_t L = 1; L < m_count; ++L)
                aL = m_L[L]->forward(aL);

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT

        return aL;
}

void network::backward(const vector &a0, const vector &dC)
{
        const auto a = new vector[m_count];
        _forward_impl(a0, a);

        this->backward(dC, a);
        delete [] a;
}

void network::backward(vector dC, const vector a[])
{
        for (int32_t L = (int32_t)m_count - 1; L > 0; --L)
                dC = m_L[L]->backward(dC, a[L]);

#ifdef BUILD_CUDA_SUPPORT
        cuda::sync(m_stream);
#endif // BUILD_CUDA_SUPPORT
}

void network::train_supervised(
        uint32_t           samples,
        const float        inputs[],
        const float        outputs[]) {

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

void network::encode(const std::filesystem::path &path) const
{
        std::ofstream file(path, std::ios::binary);
        if (!file)
                throw fatal_error("Error opening file.");

        for (uint32_t L = 0; L < m_count; ++L)
                m_L[L]->encode(file);

        file.close();
}

stream network::_create_stream([[maybe_unused]] const layer_create_info *infos) const
{
#ifdef BUILD_CUDA_SUPPORT
        static std::array<stream, 16> streams = [] {
                std::array<stream, 16> ret{};
                for (stream &str : ret)
                        str = cuda::create_stream();

                return ret;
        }();
        static uint32_t usedStreams = 0;

        uint32_t ops = 0;
        for (uint32_t i = 1; i < m_count; ++i)
                ops += infos[i - 1].count * infos[i].count;

        if (ops >= CUDA_MINIMUM) {
                const stream str = streams[usedStreams];
                usedStreams      = (usedStreams + 1) % streams.size();
                return str;
        }

#endif // BUILD_CUDA_SUPPORT
        return invalid_stream;
}

std::unique_ptr<layer> *network::_create_layers(
        const layer_create_info        *infos) const {

        auto layers = new std::unique_ptr<layer>[m_count];
        ::_create_layers(layers, m_count, infos, m_stream);

        return layers;
}

std::unique_ptr<layer> *network::_create_layers(
        const layer_create_info        *infos,
        const std::string_view         &path) const {

        namespace fs = std::filesystem;

        if (path.empty())
                return _create_layers(infos);

        if (!fs::exists(path))
                throw fatal_error("File given to Network() does not exist.");

        std::ifstream file(fs::path(path), std::ios::binary);
        auto layers = new std::unique_ptr<layer>[m_count];
        ::_create_layers(layers, m_count, infos, file, m_stream);

        file.close();
        return layers;
}

std::future<void> network::_get_supervised_future(
        const chunk        &inputs,
        const chunk        &outputs) {

        return std::async(std::launch::async, [&]() -> void {
                const size_t size = inputs.size() / this->input_size();

                for (uint32_t i = 0; i < size; ++i) {
                        const float *in  = inputs.data() + (uintptr_t)i * this->input_size();
                        const float *out = outputs.data() + (uintptr_t)i * this->output_size();

                        const auto a = new vector[m_count];
                        _forward_impl(vector(this->input_size(), in, m_stream), a);

                        const vector y(this->output_size(), out, m_stream);
                        const vector dC = 2.0f * (a[m_count - 1] - y);

                        this->backward(dC, a);
                        delete [] a;
                }
        });
}

inline void network::_forward_impl(
        const vector        &input,
        vector              a[]) const {

        a[0] = input;
        for (uint32_t L = 1; L < m_count; ++L)
                a[L] = m_L[L - 1]->forward(a[L - 1]);
}

struct _iteration_data {
        uint64_t        stateHash;
        float           reward;
        float           predicted;
        vector          policy;
        vector          *pa;
        vector          *va;

};

void network::_train_ppo(
        network            &value,
        environment        &env,
        uint32_t           epochs,
        uint32_t           max_steps) {

        std::unordered_map<uint64_t, vector> policies;
        for (uint32_t i = 0; i < epochs; ++i)
                _train_ppo_iteration(value, env, max_steps, policies);
}

void network::_train_ppo_iteration(
        network                                     &value,
        environment                                 &env,
        uint32_t                                    max_steps,
        std::unordered_map<uint64_t, vector>        &policies) {

        std::vector<_iteration_data> iteration;
        for (uint32_t s = 0, done = false; !done && s < max_steps; ++s) {
                iteration.emplace_back();
                auto &[hash, reward, predicted, policy, pa, va] = iteration.back();

                const vector state = env.get_state();
                hash               = std::hash<buf>{}(state);

                pa = new vector[m_count];
                _forward_impl(state, pa);

                va = new vector[value.m_count];
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
