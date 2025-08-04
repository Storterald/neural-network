#pragma once

#include <cstdint>
#include <limits>

#include <neural-network/utils/exceptions.h>
#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/math/math.h>
#include <neural-network/base.h>

namespace nn {

template<floating_type _type>
class activation_layer final : public layer<_type> {
public:
        using value_type = _type;

        inline activation_layer(function_type activation);

        [[nodiscard]] inline vector<value_type> forward(const vector<value_type> &input) const override;
        [[nodiscard]] inline vector<value_type> backward(const vector<value_type> &cost, const vector<value_type> &input) override;

        [[nodiscard]] inline matrix<value_type> forward(const matrix<value_type> &inputs) const override;
        [[nodiscard]] inline matrix<value_type> backward(const matrix<value_type> &costs, const matrix<value_type> &inputs) override;

        inline void encode([[maybe_unused]] std::ostream &out) const override {}
        [[nodiscard]] inline uint32_t size() const noexcept override;

private:
        function_type m_activation;

        inline void _forward_impl(uint32_t size, const buf<value_type> &input, buf<value_type> &result) const;
        inline void _backward_impl(uint32_t size, const buf<value_type> &input, const buf<value_type> &cost, buf<value_type> &result) const;

}; // class activation_layer

template<floating_type _type>
inline activation_layer<_type>::activation_layer(function_type activation) :
        m_activation(activation) {}

template<floating_type _type>
inline vector<_type> activation_layer<_type>::forward(const vector<_type> &input) const
{
        if (m_activation == none)
                return input;

        vector<_type> result(input.size(), input.stream());
        _forward_impl(input.size(), input, result);
        return result;
}

template<floating_type _type>
inline vector<_type> activation_layer<_type>::backward(const vector<_type> &cost, const vector<_type> &input)
{
        if (m_activation == none)
                return cost;

        vector<_type> result(input.size(), input.stream());
        _backward_impl(input.size(), input, cost, result);
        return result;
}

template<floating_type _type>
inline matrix<_type> activation_layer<_type>::forward(const matrix<_type> &inputs) const
{
        if (m_activation == none)
                return inputs;

        matrix<_type> result(inputs.width(), inputs.height(), inputs.stream());
        _forward_impl(result.size(), inputs, result);
        return result;
}

template<floating_type _type>
inline matrix<_type> activation_layer<_type>::backward(const matrix<_type> &costs, const matrix<_type> &inputs)
{
        if (m_activation == none)
                return costs;

        matrix<_type> result(inputs.width(), inputs.height(), inputs.stream());
        _backward_impl(result.size(), inputs, costs, result);
        return result;
}

template<floating_type _type>
inline uint32_t activation_layer<_type>::size() const noexcept
{
        return std::numeric_limits<uint32_t>::max();
}

template<floating_type _type>
inline void activation_layer<_type>::_forward_impl(uint32_t size, const buf<_type> &input, buf<_type> &result) const
{
        switch (m_activation) {
        case tanh:
                math::tanh(size, input, result);
                break;
        case relu:
                math::ReLU(size, input, result);
                break;
        case none:
        default:
                throw fatal_error("Activation function not implemented.");
        }
}

template<floating_type _type>
inline void activation_layer<_type>::_backward_impl(uint32_t size, const buf<_type> &input, const buf<_type> &cost, buf<_type> &result) const
{
        switch (m_activation) {
        case tanh:
                math::tanh_derivative(size, input, result);
                break;
        case relu:
                math::ReLU_derivative(size, input, result);
                break;
        case none:
        default:
                throw fatal_error("Activation function not implemented.");
        }

        math::mul(input.size(), result, cost, result);
}

} // namespace nn
