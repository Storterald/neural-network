#include <neural-network/math/Math.h>

#include <type_traits>
#include <concepts>
#include <cstdint>

#include <neural-network/intrinsic/Intrinsic.h>
#include <neural-network/utils/Macros.h>
#include <neural-network/types/Data.h>
#include <neural-network/math/_Math.h>
#include <neural-network/Base.h>

template<typename T>
static constexpr auto _get(T &&v, NN Data::DataLocation location)
{
        if constexpr (std::same_as<std::remove_cvref_t<T>, NN Data>)
                return v.as_span(location, true);
        else
                return v;
}

#define GET_ALL(__dest__, ...)                          \
__VA_OPT__(EXPAND(__GET_ALL(__dest__, __VA_ARGS__)))

#define __GET_ALL(__dest__, __name__, ...)                                      \
_get(__name__, __dest__) __VA_OPT__(, __GET_ALL2 PARENS (__dest__, __VA_ARGS__))

#define __GET_ALL2() __GET_ALL

#ifdef BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA_MINIMUM(__name__, __size__, ...)                          \
if ((__size__) >= CUDA_MINIMUM)                                                 \
        return _Math<MATH_CUDA>:: __name__ (GET_ALL(Data::DEVICE, __VA_ARGS__))
#else // BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA_MINIMUM(...)
#endif // BUILD_CUDA_SUPPORT

#define SIMD_SWITCH(__name__, ...)                                                              \
do {                                                                                            \
        switch (Intrinsic::support()) {                                                         \
        case SIMD_AVX512:                                                                       \
                return _Math<MATH_AVX512>:: __name__ (GET_ALL(Data::HOST, __VA_ARGS__));        \
        case SIMD_AVX:                                                                          \
                return _Math<MATH_AVX>:: __name__ (GET_ALL(Data::HOST, __VA_ARGS__));           \
        case SIMD_SSE3:                                                                         \
                return _Math<MATH_SSE3>:: __name__ (GET_ALL(Data::HOST, __VA_ARGS__));          \
        case SIMD_UNSUPPORTED:                                                                  \
                return _Math<MATH_NORMAL>:: __name__ (GET_ALL(Data::HOST, __VA_ARGS__));        \
        }                                                                                       \
} while (false)

#define DECLARE_MATH_FUNCTION(__name__, __size__, ...)                          \
void Math:: __name__ (GET_ARGS(__VA_ARGS__))                                    \
{                                                                               \
        CHECK_IF_CUDA_MINIMUM(__name__, __size__, GET_ARGS_NAMES(__VA_ARGS__)); \
        SIMD_SWITCH(__name__, GET_ARGS_NAMES(__VA_ARGS__));                     \
}

NN_BEGIN

DECLARE_MATH_FUNCTION(sum, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        Data &,              result)

DECLARE_MATH_FUNCTION(sub, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        Data &,              result)

DECLARE_MATH_FUNCTION(mul, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        Data &,              result)

DECLARE_MATH_FUNCTION(div, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        Data &,              result)

DECLARE_MATH_FUNCTION(sum, size,
        uint32_t,            size,
        const Data &,        data,
        float,               scalar,
        Data &,              result)

DECLARE_MATH_FUNCTION(sub, size,
        uint32_t,            size,
        const Data &,        data,
        float,               scalar,
        Data &,              result)

DECLARE_MATH_FUNCTION(mul, size,
        uint32_t,            size,
        const Data &,        data,
        float,               scalar,
        Data &,              result)

DECLARE_MATH_FUNCTION(div, size,
        uint32_t,            size,
        const Data &,        data,
        float,               scalar,
        Data &,              result)

DECLARE_MATH_FUNCTION(tanh, size,
        uint32_t,            size,
        const Data &,        data,
        Data &,              result)

DECLARE_MATH_FUNCTION(tanh_derivative, size,
        uint32_t,            size,
        const Data &,        data,
        Data &,              result)

DECLARE_MATH_FUNCTION(ReLU, size,
        uint32_t,            size,
        const Data &,        data,
        Data &,              result)

DECLARE_MATH_FUNCTION(ReLU_derivative, size,
        uint32_t,            size,
        const Data &,        data,
        Data &,              result)

DECLARE_MATH_FUNCTION(min, size,
        uint32_t,            size,
        const Data &,        data,
        float,               min,
        Data &,              result)

DECLARE_MATH_FUNCTION(max, size,
        uint32_t,            size,
        const Data &,        data,
        float,               max,
        Data &,              result)

DECLARE_MATH_FUNCTION(clamp, size,
        uint32_t,            size,
        const Data &,        data,
        float,               min,
        float,               max,
        Data &,              result)

DECLARE_MATH_FUNCTION(min, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        Data &,              result)

DECLARE_MATH_FUNCTION(max, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        Data &,              result)

DECLARE_MATH_FUNCTION(clamp, size,
        uint32_t,            size,
        const Data &,        data,
        const Data &,        min,
        const Data &,        max,
        Data &,              result)

DECLARE_MATH_FUNCTION(compare, size,
        uint32_t,            size,
        const Data &,        first,
        const Data &,        second,
        bool *,              result)

DECLARE_MATH_FUNCTION(matvec_mul, width * height,
        uint32_t,            width,
        uint32_t,            height,
        const Data &,        matrix,
        const Data &,        vector,
        Data &,              result)

NN_END
