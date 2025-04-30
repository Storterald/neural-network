#pragma once

#include <utility>

#include <neural-network/types/vector.h>

namespace nn {

class environment {
public:
        [[nodiscard]] virtual vector getState() const = 0;

        virtual std::pair<float, bool> step(const vector &action) = 0;

        virtual void reset() = 0;

        virtual ~environment() = default;

}; // interface environment

} // namespace nn
