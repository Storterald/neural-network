#pragma once

#include <iostream>
#include <cstdint>
#include <random>
#include <mutex>

#include <neural-network/network/layer.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

template<floating_type _type>
class dense_layer final : public layer<_type> {
public:
        using value_type = _type;

        dense_layer(
                uint32_t prev,
                uint32_t size,
                stream_t stream = invalid_stream);

        dense_layer(
                uint32_t     prev,
                uint32_t     size,
                std::istream &in,
                stream_t     stream = invalid_stream);

        [[nodiscard]] inline vector<value_type> forward(const vector<value_type> &input) const override;
        [[nodiscard]] inline vector<value_type> backward(const vector<value_type> &cost, const vector<value_type> &input) override;

        [[nodiscard]] inline matrix<value_type> forward(const matrix<value_type> &inputs) const override;
        [[nodiscard]] inline matrix<value_type> backward(const matrix<value_type> &costs, const matrix<value_type> &inputs) override;

        void encode(std::ostream &out) const override;
        [[nodiscard]] inline uint32_t size() const noexcept override;

private:
        stream_t           m_stream;
        matrix<value_type> m_w;
        vector<value_type> m_b;
        std::mutex         m_mutex;

}; // class dense_layer

template<floating_type _type>
dense_layer<_type>::dense_layer(
        uint32_t prev,
        uint32_t size,
        stream_t stream) :
                m_stream(stream),
                m_w(prev, size, stream),
                m_b(size, stream),
                m_mutex() {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        span tmp = m_w.data(loc_type::host, true);
        for (uint32_t i = 0; i < m_w.height(); i++)
                for (uint32_t j = 0; j < m_w.width(); j++)
                        tmp[i * m_w.width() + j] = dis(gen);
}

template<floating_type _type>
dense_layer<_type>::dense_layer(
        uint32_t     prev,
        uint32_t     size,
        std::istream &in,
        stream_t     stream) :
                m_stream(stream),
                m_w(prev, size, stream),
                m_b(size, stream),
                m_mutex() {

        const auto wsize = static_cast<std::streamsize>(m_w.size() * sizeof(float));
        in.read((char *)(float *)m_w.data(loc_type::host, true), wsize);
        const auto bsize = static_cast<std::streamsize>(m_b.size() * sizeof(float));
        in.read((char *)(float *)m_b.data(loc_type::host, true), bsize);
}

template<floating_type _type>
vector<_type> dense_layer<_type>::forward(const vector<_type> &input) const
{
        return m_w * input + m_b;
}

template<floating_type _type>
vector<_type> dense_layer<_type>::backward(const vector<_type> &db, const vector<_type> &input)
{
        matrix<_type> dw(m_w.width(), m_w.height(), m_stream);
        vector<_type> cost(m_w.width(), m_stream);

        (void)db;
        (void)input;

        // for (uint32_t j = 0; j < m_w.height(); ++j) {
        //         const uint32_t idx = j * m_w.width();
        //         math::mul(m_w.width(), input, db[j], &dw[idx]);
        //         math::fma(m_w.width(), &mw[idx], db[j], result, result);
        // }

        {
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                m_b -= db * LEARNING_RATE;
        }

        return cost;
}

template<floating_type _type>
matrix<_type> dense_layer<_type>::forward(const matrix<_type> &inputs) const
{
        return inputs * m_w.transposed_view() + m_b; // TODO single cuda kernel
}

template<floating_type _type>
matrix<_type> dense_layer<_type>::backward(const matrix<_type> &dbs, const matrix<_type> &inputs)
{
        // TODO single cuda kernel

        const matrix dw   = dbs.transposed_view() * inputs;
        const matrix cost = dbs * m_w;

        {
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                // TODO m_b -= db * LEARNING_RATE;
        }

        return cost;
}

template<floating_type _type>
void dense_layer<_type>::encode(std::ostream &out) const
{
        const auto wsize = static_cast<std::streamsize>(m_w.size() * sizeof(float));
        out.write((char *)(const float *)m_w.data(loc_type::host), wsize);
        const auto bsize = static_cast<std::streamsize>(m_b.size() * sizeof(float));
        out.write((char *)(const float *)m_b.data(loc_type::host),bsize);
}

template<floating_type _type>
inline uint32_t dense_layer<_type>::size() const noexcept
{
        return m_b.size();
}

} // namespace nn
