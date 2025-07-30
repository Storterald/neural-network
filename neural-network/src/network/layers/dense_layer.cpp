#include <neural-network/network/layers/dense_layer.h>

#include <cstdint>
#include <random>
#include <mutex>

#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/buf.h>
#include <neural-network/base.h>

#include "../../math/_math_cpu.h"

namespace nn {

dense_layer::dense_layer(
        uint32_t             prev,
        uint32_t             size,
        stream               stream) :

        m_stream(stream),
        m_w(prev, size, stream),
        m_b(size, stream),
        m_mutex() {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        span tmp = m_w.data(buf::host, true);
        for (uint32_t i = 0; i < m_w.height(); i++)
                for (uint32_t j = 0; j < m_w.width(); j++)
                        tmp[i * m_w.width() + j] = dis(gen);
}

dense_layer::dense_layer(
        uint32_t             prev,
        uint32_t             size,
        std::istream         &in,
        stream               stream) :

        m_stream(stream),
        m_w(prev, size, stream),
        m_b(size, stream),
        m_mutex() {

        const auto wsize = static_cast<std::streamsize>(m_w.size() * sizeof(float));
        in.read((char *)(float *)m_w.data(buf::host, true), wsize);
        const auto bsize = static_cast<std::streamsize>(m_b.size() * sizeof(float));
        in.read((char *)(float *)m_b.data(buf::host, true), bsize);
}

vector dense_layer::forward(const vector &input) const
{
        return m_w * input + m_b;
}

vector dense_layer::backward(const vector &db, const vector &input)
{
        matrix dw(m_w.width(), m_w.height(), m_stream);
        vector cost(m_w.width(), m_stream);

#ifdef BUILD_CUDA_SUPPORT
        if (m_w.location() == buf::host) {
#endif // BUILD_CUDA_SUPPORT
                _cpu_backward(
                        input.view(buf::host), dw.data(buf::host, true),
                        db.view(buf::host),    cost.data(buf::host, true));
#ifdef BUILD_CUDA_SUPPORT
        } else {
                _gpu_backward(
                        input.view(buf::device), dw.data(buf::device, true),
                        db.view(buf::device),    cost.data(buf::device, true));
        }
#endif // BUILD_CUDA_SUPPORT

        {
                std::lock_guard lock(m_mutex);

                m_w -= dw * LEARNING_RATE;
                m_b -= db * LEARNING_RATE;
        }

        return cost;
}

void dense_layer::encode(std::ostream &out) const
{
        const auto wsize = static_cast<std::streamsize>(m_w.size() * sizeof(float));
        out.write((char *)(const float *)m_w.view(buf::loc_type::host), wsize);
        const auto bsize = static_cast<std::streamsize>(m_b.size() * sizeof(float));
        out.write((char *)(const float *)m_b.view(buf::loc_type::host),bsize);
}

void dense_layer::_cpu_backward(
        const float        input[],
        float              dw[],
        const float        db[],
        float              result[]) const {

        const float *mw = m_w.begin().get();

        for (uint32_t j = 0; j < m_w.height(); ++j) {
                const uint32_t idx = j * m_w.width();
                _math_cpu::mul(m_w.width(), input, db[j], &dw[idx]);
                _math_cpu::fma(m_w.width(), &mw[idx], db[j], result, result);
        }
}

} // namespace nn
