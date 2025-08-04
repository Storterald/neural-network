#pragma once

#include <ostream>
#include <cstdint>

#include <neural-network/types/vector.h>
#include <neural-network/types/matrix.h>
#include <neural-network/base.h>

namespace nn {

enum function_type : uint16_t {
        none,
        tanh,
        relu

}; // enum function_type

enum layer_type : uint16_t {
        input,
        dense,
        activation

}; // enum layer_type

struct layer_create_info {
        layer_type    type;
        uint32_t      count      = 0;
        function_type activation = function_type::none;

}; // struct layer_create_info

template<floating_type _type>
class layer {
public:
        [[nodiscard]] virtual vector<_type> forward(const vector<_type> &input) const = 0;
        virtual vector<_type> backward(const vector<_type> &cost, const vector<_type> &input) = 0;

        [[nodiscard]] virtual matrix<_type> forward(const matrix<_type> &inputs) const = 0;
        virtual matrix<_type> backward(const matrix<_type> &costs, const matrix<_type> &inputs) = 0;

        virtual void encode(std::ostream &file) const = 0;
        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~layer() = default;

}; // interface layer

} // namespace nn
