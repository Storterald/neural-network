#include <neural-network/network/layers/activation_layer.h>

#include <neural-network/network/layer.h>
#include <neural-network/types/vector.h>
#include <neural-network/math/math.h>

namespace nn {

vector activation_layer::forward(const vector &input) const
{
        if (m_activation == none)
                return input;

        vector result(input.size(), input.stream());
        switch (m_activation) {
        case tanh:
                math::tanh(input.size(), input, result);
                break;
        case relu:
                math::ReLU(input.size(), input, result);
                break;
        case none:
        default:
                throw fatal_error("Activation function not implemented.");
        }

        return result;
}

vector activation_layer::backward(const vector &cost, const vector &input)
{
        if (m_activation == none)
                return cost;

        vector result(input.size(), input.stream());
        switch (m_activation) {
        case tanh:
                math::tanh_derivative(input.size(), input, result);
                break;
        case relu:
                math::ReLU_derivative(input.size(), input, result);
                break;
        case none:
        default:
                throw fatal_error("Activation function not implemented.");
        }

        return result * cost;
}

} // namespace nn