#pragma once

#include "Vector.cuh"

namespace Fast {

        Vector relu(const Vector &vec);
        Vector reluDerivative(const Vector &vec);
        Vector tanh(const Vector &vec);
        Vector tanhDerivative(const Vector &vec);

} // namespace Fast