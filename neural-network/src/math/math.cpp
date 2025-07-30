#include <neural-network/math/math.h>

#include <type_traits>
#include <cstdint>

#include <neural-network/utils/macros.h>
#include <neural-network/types/buf.h>

#include "_math_cpu.h"

#ifdef BUILD_CUDA_SUPPORT
#include "_math_cuda.h"
#endif // BUILD_CUDA_SUPPORT

template<typename T>
static constexpr decltype(auto) _get(T &&v, nn::buf::loc_type location)
{
        using raw = std::remove_cvref_t<T>;

        if constexpr (std::is_same_v<raw, nn::buf>) {
                if constexpr (std::is_const_v<std::remove_reference_t<T>>)
                        return std::forward<T>(v).view(location);
                else
                        return std::forward<T>(v).data(location, true);
        } else {
                return std::forward<T>(v);
        }
}

#define GET_ALL(__dest__, ...)                          \
__VA_OPT__(EXPAND(__GET_ALL(__dest__, __VA_ARGS__)))

#define __GET_ALL(__dest__, __name__, ...)                                      \
_get(__name__, __dest__) __VA_OPT__(, __GET_ALL2 PARENS (__dest__, __VA_ARGS__))

#define __GET_ALL2() __GET_ALL

#ifdef BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA(__name__, __first__, ...)                                                                         \
do {                                                                                                                    \
        if ((__first__).location() == buf::loc_type::device)                                                            \
                return _math_cuda:: __name__ (GET_ALL(buf::loc_type::device, __VA_ARGS__), (__first__).stream());       \
} while (false)
#else // BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA(...)
#endif // BUILD_CUDA_SUPPORT

#define DECLARE_MATH_FUNCTION(__name__, __stream__, ...)                                                \
void math:: __name__ (GET_ARGS(__VA_ARGS__))                                                            \
{                                                                                                       \
        CHECK_IF_CUDA(__name__, __stream__, GET_ARGS_NAMES(__VA_ARGS__));                               \
        return _math_cpu:: __name__ (GET_ALL(buf::loc_type::host, GET_ARGS_NAMES(__VA_ARGS__)));        \
}

namespace nn {

DECLARE_MATH_FUNCTION(sum, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(sub, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(mul, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(div, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(fma, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        const buf &,        third,
        buf &,              result)

DECLARE_MATH_FUNCTION(sum, data,
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(sub, data,
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(mul, data,
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(div, data,
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(fma, first,
        uint32_t,           size,
        const buf &,        first,
        float,              scalar,
        const buf &,        third,
        buf &,              result)
        
DECLARE_MATH_FUNCTION(tanh, data,
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(tanh_derivative, data,
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(ReLU, data,
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(ReLU_derivative, data,
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(min, data,
        uint32_t,           size,
        const buf &,        data,
        float,              min,
        buf &,              result)

DECLARE_MATH_FUNCTION(max, data,
        uint32_t,           size,
        const buf &,        data,
        float,              max,
        buf &,              result)

DECLARE_MATH_FUNCTION(clamp, data,
        uint32_t,           size,
        const buf &,        data,
        float,              min,
        float,              max,
        buf &,              result)

DECLARE_MATH_FUNCTION(min, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(max, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(clamp, data,
        uint32_t,           size,
        const buf &,        data,
        const buf &,        min,
        const buf &,        max,
        buf &,              result)

DECLARE_MATH_FUNCTION(compare, first,
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        bool *,             result)

DECLARE_MATH_FUNCTION(matvec_mul, matrix,
        uint32_t,           width,
        uint32_t,           height,
        const buf &,        matrix,
        const buf &,        vector,
        buf &,              result)

} // namespace nn
