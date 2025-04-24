#include "Math.h"

#include "../intrinsic/Intrinsic.h"
#include "_Math.h"
#include "Base.h"

#ifdef BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA_MINIMUM(__size__, __foo__, ...)   \
if ((__size__) >= CUDA_MINIMUM)                         \
        return _Math<MATH_CUDA>:: __foo__ (__VA_ARGS__)
#else // BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA_MINIMUM(...)
#endif // BUILD_CUDA_SUPPORT

#define DECLARE_MATH_FUNCTION(__name__, __size__, ...)                                  \
void Math:: __name__ (GET_ARGS(__VA_ARGS__))                                            \
{                                                                                       \
        CHECK_IF_CUDA_MINIMUM(__size__, __name__, GET_ARGS_NAMES(__VA_ARGS__));         \
                                                                                        \
        switch(Intrinsic::support()) {                                                  \
        case SIMD_AVX512:                                                               \
                return _Math<MATH_AVX512>:: __name__ (GET_ARGS_NAMES(__VA_ARGS__));     \
        case SIMD_AVX:                                                                  \
                return _Math<MATH_AVX>:: __name__ (GET_ARGS_NAMES(__VA_ARGS__));        \
        case SIMD_SSE:                                                                  \
                return _Math<MATH_SSE>:: __name__ (GET_ARGS_NAMES(__VA_ARGS__));        \
        case SIMD_UNSUPPORTED:                                                          \
                return _Math<MATH_NORMAL>:: __name__ (GET_ARGS_NAMES(__VA_ARGS__));     \
        }                                                                               \
}

DECLARE_MATH_FUNCTION(sum, size, uint32_t, size, const float *, first, const float*, second, float *, result)
DECLARE_MATH_FUNCTION(sub, size, uint32_t, size, const float *, first, const float *, second, float *, result)
DECLARE_MATH_FUNCTION(mul, size, uint32_t, size, const float *, first, const float *, second, float *, result)
DECLARE_MATH_FUNCTION(div, size, uint32_t, size, const float *, first, const float *, second, float *, result)
DECLARE_MATH_FUNCTION(sum, size, uint32_t, size, const float *, first, float, scalar, float *, result)
DECLARE_MATH_FUNCTION(sub, size, uint32_t, size, const float *, first, float, scalar, float *, result)
DECLARE_MATH_FUNCTION(mul, size, uint32_t, size, const float *, first, float, scalar, float *, result)
DECLARE_MATH_FUNCTION(div, size, uint32_t, size, const float *, first, float, scalar, float *, result)
DECLARE_MATH_FUNCTION(tanh, size, uint32_t, size, const float *, data, float *, result)
DECLARE_MATH_FUNCTION(tanh_derivative, size, uint32_t, size, const float *, data, float *, result)
DECLARE_MATH_FUNCTION(ReLU, size, uint32_t, size, const float *, data, float *, result)
DECLARE_MATH_FUNCTION(ReLU_derivative, size, uint32_t, size, const float *, data, float *, result)
DECLARE_MATH_FUNCTION(min, size, uint32_t, size, const float *, first, const float *, second, float *, result)
DECLARE_MATH_FUNCTION(max, size, uint32_t, size, const float *, first, const float *, second, float *, result)
DECLARE_MATH_FUNCTION(clamp, size, uint32_t, size, const float *, data, const float *, min, const float *, max, float *, result)
DECLARE_MATH_FUNCTION(min, size, uint32_t, size, const float *, data, float, min, float *, result)
DECLARE_MATH_FUNCTION(max, size, uint32_t, size, const float *, data, float, max, float *, result)
DECLARE_MATH_FUNCTION(clamp, size, uint32_t, size, const float *, data, float, min, float, max, float *, result)
DECLARE_MATH_FUNCTION(matvec_mul, width * height, uint32_t, width, uint32_t, height, const float *, matrix, const float *, vector, float *, result)