#pragma once

#include <iostream>
#include <cstdint>
#include <memory>

#include <neural-network/types/vector.h>
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

class layer {
public:
        static std::unique_ptr<layer> create(
                uint32_t                       prev,
                const layer_create_info        &info,
                stream                         stream = invalid_stream);

        static std::unique_ptr<layer> create(
                uint32_t                       prev,
                const layer_create_info        &info,
                std::istream                   &in,
                stream                         stream = invalid_stream);

        [[nodiscard]] virtual vector forward(const vector &input) const = 0;
        virtual vector backward(const vector &cost, const vector &input) = 0;

        virtual void encode(std::ostream &file) const = 0;

        [[nodiscard]] virtual uint32_t size() const = 0;

        virtual ~layer() = default;

}; // interface layer

} // namespace nn
