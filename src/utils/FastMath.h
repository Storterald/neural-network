#pragma once

#include <cmath>

#include "../math/Vector.h"

namespace Fast {

        constexpr float relu(float x)
        {
                return std::max(x, 0.0f);
        }

        constexpr float reluDerivative(float x)
        {
                return x >= 0.0f ? 1.0f : 0.0f;
        }

        inline float tanh(float x)
        {
                // Around 6-8 times faster than std::tanh in release mode
                // and around 40% in debug mode (time is manly used by writing in the console)

                // Approximate tanh(x) using a polynomial for stability and speed
                // For |x| > 6, tanh(x) is approximately 1 or -1
                if (std::abs(x) >= 6.0f)
                        return std::copysign(1.0f, x);

                // 7th-order approximation (accurate within 0.000405).
                // https://www.desmos.com/calculator/2myik1oe4x
                const float x2 { x * x };
                return  x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) / (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
        }

        inline float tanhDerivative(float x)
        {
                // Uses the tanhFast to calculate the derivative way faster than
                // with using std::tanh, also more accurate than tanhFast.

                // tanh'(0) = 1
                if (x == 0.0f)
                        return 1.0f;

                // After this point, the approximation of the derivative goes negative.
                // Instead, the tanh' goes closer and closer to 0 (accurate within 0.000170).
                // https://www.desmos.com/calculator/2myik1oe4x
                if (std::abs(x) > 4.9f)
                        return 0.0f;

                const float tanh { Fast::tanh(x) };
#ifdef DEBUG_MODE_ENABLED
                const float value { 1.0f - tanh * tanh };
                if (value < 0.0f)
                        throw Logger::fatal_error("Returned value from tanhDerivativeFast is negative.");

                return value;
#else
                return 1.0f - tanh * tanh;
#endif // DEBUG_MODE_ENABLED
        }

        Vector relu(const Vector &vec);
        Vector reluDerivative(const Vector &vec);
        Vector tanh(const Vector &vec);
        Vector tanhDerivative(const Vector &vec);

}