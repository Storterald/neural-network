#pragma once

#include <utility>

#include <neural-network/types/vector.h>
#include <neural-network/base.h>

namespace nn {

template<floating_type _type>
class environment {
public:
        using value_type = _type;

        [[nodiscard]] virtual vector<value_type> get_state() const = 0;

        virtual std::pair<float, bool> step(const vector<value_type> &action) = 0;

        virtual void reset() = 0;

        virtual ~environment() = default;

}; // interface environment

} // namespace nn
