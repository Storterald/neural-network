#pragma once

#include <utility>

#ifdef USE_CUDA
#include "../../cuda/Vector.h"
#else
#include "../../math/Vector.h"
#endif // USE_CUDA

class IEnvironment {
public:
        [[nodiscard]] virtual Vector getState() const = 0;
        virtual std::pair<float, bool> step(const Vector &action) = 0;
        virtual void reset() = 0;
        virtual ~IEnvironment() = default;

}; // interface IEnvironment