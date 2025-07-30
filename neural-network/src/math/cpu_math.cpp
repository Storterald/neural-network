#include "_math_cpu.h"

#include <cstdint>

#include <neural-network/utils/macros.h>

#ifdef TARGET_X86_64
#include "_math_simd.h"
#else // TARGET_X86_64
#include "_math_normal.h"
#endif // !TARGET_X86_64

#ifdef TARGET_X86_64
#define RUN(__name__, ...)                                      \
do {                                                            \
        _math_simd:: __name__ (GET_ARGS_NAMES(__VA_ARGS__));    \
} while (false)
#else // TARGET_X86_64
#define RUN(__name__, ...)                                      \
do {                                                            \
        _math_normal:: __name__ (GET_ARGS_NAMES(__VA_ARGS__));  \
} while (false)
#endif // TARGET_X86_64

#define DECLARE_CPU_FUNCTION(__name__, __size__, ...)   \
void _math_cpu:: __name__ (GET_ARGS(__VA_ARGS__))       \
{                                                       \
        RUN(__name__, __VA_ARGS__);                     \
}

namespace nn {

DECLARE_CPU_FUNCTION(sum, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CPU_FUNCTION(sub, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CPU_FUNCTION(mul, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CPU_FUNCTION(div, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CPU_FUNCTION(fma, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        const float *,        third,
        float *,              result)

DECLARE_CPU_FUNCTION(sum, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CPU_FUNCTION(sub, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CPU_FUNCTION(mul, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CPU_FUNCTION(div, size,
        uint32_t,             size,
        const float *,        data,
        float,                scalar,
        float *,              result)

DECLARE_CPU_FUNCTION(fma, size,
        uint32_t,             size,
        const float *,        first,
        float,                scalar,
        const float *,        third,
        float *,              result)

DECLARE_CPU_FUNCTION(tanh, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CPU_FUNCTION(tanh_derivative, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CPU_FUNCTION(ReLU, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CPU_FUNCTION(ReLU_derivative, size,
        uint32_t,             size,
        const float *,        data,
        float *,              result)

DECLARE_CPU_FUNCTION(min, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CPU_FUNCTION(max, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        float *,              result)

DECLARE_CPU_FUNCTION(clamp, size,
        uint32_t,             size,
        const float *,        data,
        const float *,        min,
        const float *,        max,
        float *,              result)

DECLARE_CPU_FUNCTION(min, size,
        uint32_t,             size,
        const float *,        data,
        float,                min,
        float *,              result)

DECLARE_CPU_FUNCTION(max, size,
        uint32_t,             size,
        const float *,        data,
        float,                max,
        float *,              result)

DECLARE_CPU_FUNCTION(clamp, size,
        uint32_t,             size,
        const float *,        data,
        float,                min,
        float,                max,
        float *,              result)

DECLARE_CPU_FUNCTION(compare, size,
        uint32_t,             size,
        const float *,        first,
        const float *,        second,
        bool *,               result)

DECLARE_CPU_FUNCTION(matvec_mul, width,
        uint32_t,             width,
        uint32_t,             height,
        const float *,        matrix,
        const float *,        vector,
        float *,              result)

} // namespace nn
