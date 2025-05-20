#include <neural-network/math/math.h>

#include <type_traits>
#include <cstdint>

#include <neural-network/intrinsic/intrinsic.h>
#include <neural-network/utils/macros.h>
#include <neural-network/types/buf.h>
#include "_math.h"

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
#define CHECK_IF_CUDA_MINIMUM(__name__, __size__, __stream__, ...)                              \
if ((__size__) >= CUDA_MINIMUM)                                                                 \
        return _math<MATH_CUDA>:: __name__ (GET_ALL(buf::DEVICE, __VA_ARGS__), __stream__)
#else // BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA_MINIMUM(...)
#endif // BUILD_CUDA_SUPPORT

#define SIMD_SWITCH(__name__, ...)                                                              \
do {                                                                                            \
        switch (intrinsic::support()) {                                                         \
        case SIMD_AVX512:                                                                       \
                return _math<MATH_AVX512>:: __name__ (GET_ALL(buf::HOST, __VA_ARGS__));         \
        case SIMD_AVX:                                                                          \
                return _math<MATH_AVX>:: __name__ (GET_ALL(buf::HOST, __VA_ARGS__));            \
        case SIMD_SSE3:                                                                         \
                return _math<MATH_SSE3>:: __name__ (GET_ALL(buf::HOST, __VA_ARGS__));           \
        case SIMD_UNSUPPORTED:                                                                  \
                return _math<MATH_NORMAL>:: __name__ (GET_ALL(buf::HOST, __VA_ARGS__));         \
        }                                                                                       \
} while (false)

#define DECLARE_MATH_FUNCTION(__name__, __size__, __stream__, ...)                              \
void math:: __name__ (GET_ARGS(__VA_ARGS__))                                                    \
{                                                                                               \
        CHECK_IF_CUDA_MINIMUM(__name__, __size__, __stream__, GET_ARGS_NAMES(__VA_ARGS__));     \
        SIMD_SWITCH(__name__, GET_ARGS_NAMES(__VA_ARGS__));                                     \
}

namespace nn {

DECLARE_MATH_FUNCTION(sum, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(sub, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(mul, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(div, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(sum, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(sub, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(mul, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(div, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              scalar,
        buf &,              result)

DECLARE_MATH_FUNCTION(tanh, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(tanh_derivative, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(ReLU, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(ReLU_derivative, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        buf &,              result)

DECLARE_MATH_FUNCTION(min, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              min,
        buf &,              result)

DECLARE_MATH_FUNCTION(max, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              max,
        buf &,              result)

DECLARE_MATH_FUNCTION(clamp, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        float,              min,
        float,              max,
        buf &,              result)

DECLARE_MATH_FUNCTION(min, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(max, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        buf &,              result)

DECLARE_MATH_FUNCTION(clamp, size, data.stream(),
        uint32_t,           size,
        const buf &,        data,
        const buf &,        min,
        const buf &,        max,
        buf &,              result)

DECLARE_MATH_FUNCTION(compare, size, first.stream(),
        uint32_t,           size,
        const buf &,        first,
        const buf &,        second,
        bool *,             result)

DECLARE_MATH_FUNCTION(matvec_mul, width * height, matrix.stream(),
        uint32_t,           width,
        uint32_t,           height,
        const buf &,        matrix,
        const buf &,        vector,
        buf &,              result)

} // namespace nn
