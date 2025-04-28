#pragma once

#include <utility>

#include <neural-network/types/Vector.h>
#include <neural-network/Base.h>

NN_BEGIN

class IEnvironment {
public:
        [[nodiscard]] virtual Vector getState() const = 0;

        virtual std::pair<float, bool> step(const Vector &action) = 0;

        virtual void reset() = 0;

        virtual ~IEnvironment() = default;

}; // interface IEnvironment

NN_END
