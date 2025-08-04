#pragma once

#include <cstdint>

#include <neural-network/utils/exceptions.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

template<floating_type _type>
class input_layer final : public layer<_type> {
public:
        using value_type = _type;

        inline input_layer(uint32_t size);

        [[nodiscard]] inline vector<value_type> forward(const vector<value_type> &input) const override;
        inline vector<value_type> backward(const vector<value_type> &cost, const vector<value_type> &input) override;

        [[nodiscard]] inline matrix<value_type> forward(const matrix<value_type> &inputs) const override;
        inline matrix<value_type> backward(const matrix<value_type> &costs, const matrix<value_type> &inputs) override;

        inline void encode([[maybe_unused]] std::ostream &out) const override {}
        [[nodiscard]] inline uint32_t size() const noexcept override;

private:
        uint32_t m_size;

}; // class input_layer

template<floating_type _type>
inline input_layer<_type>::input_layer(uint32_t size) :
        m_size(size) {}

template<floating_type _type>
inline vector<_type> input_layer<_type>::forward(const vector<_type> &input) const
{
        return input;
}

template<floating_type _type>
inline vector<_type> input_layer<_type>::backward(const vector<_type> &, const vector<_type> &)
{
        throw fatal_error("Cannot call backward on an input layer");
}

template<floating_type _type>
inline matrix<_type> input_layer<_type>::forward(const matrix<_type> &inputs) const
{
        return inputs;
}

template<floating_type _type>
inline matrix<_type> input_layer<_type>::backward(const matrix<_type> &, const matrix<_type> &)
{
        throw fatal_error("Cannot call backward on an input layer");
}

template<floating_type _type>
inline uint32_t input_layer<_type>::size() const noexcept
{
        return m_size;
}

} // namespace nn
