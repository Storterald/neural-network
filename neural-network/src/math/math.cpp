#include <neural-network/math/math.h>

#include <type_traits>
#include <cstdint>

#include <neural-network/utils/macros.h>
#include <neural-network/types/buf.h>

#include "_math_cpu.h"

#ifdef BUILD_CUDA_SUPPORT
#include "_math_cuda.h"
#endif // BUILD_CUDA_SUPPORT

static constexpr uint32_t THRESHOLD = 20000;

template<typename _type, typename T>
static constexpr decltype(auto) _get(T &&v, nn::loc_type location)
{
        using raw = std::remove_cvref_t<T>;

        if constexpr (std::is_same_v<raw, nn::buf<_type>>) {
                if constexpr (std::is_const_v<std::remove_reference_t<T>>)
                        return std::forward<T>(v).data(location);
                else
                        return std::forward<T>(v).data(location, true);
        } else {
                return std::forward<T>(v);
        }
}

#define GET_ALL(__type__, __dest__, ...)                       \
__VA_OPT__(EXPAND(__GET_ALL(__type__, __dest__, __VA_ARGS__)))

#define __GET_ALL(__type__, __dest__, __name__, ...)                                                    \
_get<__type__>(__name__, __dest__) __VA_OPT__(, __GET_ALL2 PARENS (__type__, __dest__, __VA_ARGS__))

#define __GET_ALL2() __GET_ALL

#ifdef BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA(__name__, __type__, __first__, ...)                                                               \
do {                                                                                                                    \
        if ((__first__).location() == loc_type::device)                                                                 \
                return _math_cuda:: __name__ (GET_ALL(__type__, loc_type::device, __VA_ARGS__), (__first__).stream());  \
} while (false)
#else // BUILD_CUDA_SUPPORT
#define CHECK_IF_CUDA(...)
#endif // BUILD_CUDA_SUPPORT

#define DECLARE_MATH_FUNCTION(__name__, __type__, __stream__, __size__, ...)                                                    \
template<> void math:: __name__<__type__> (GET_ARGS(__VA_ARGS__))                                                               \
{                                                                                                                               \
        CHECK_IF_CUDA(__name__, __type__, __stream__, GET_ARGS_NAMES(__VA_ARGS__));                                             \
        if ((__size__) >= THRESHOLD)                                                                                            \
                return _math_cpu:: __name__ (nn::parallel, GET_ALL(__type__, loc_type::host, GET_ARGS_NAMES(__VA_ARGS__)));     \
        return _math_cpu:: __name__ (GET_ALL(__type__, loc_type::host, GET_ARGS_NAMES(__VA_ARGS__)));                           \
}

namespace nn {

DECLARE_MATH_FUNCTION(sum, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(sub, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(mul, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(div, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(fma, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        const buf<float> &, third,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(sum, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              scalar,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(sub, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              scalar,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(mul, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              scalar,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(div, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              scalar,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(fma, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        float,              scalar,
        const buf<float> &, third,
        buf<float> &,       result)
        
DECLARE_MATH_FUNCTION(tanh, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(tanh_derivative, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(ReLU, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(ReLU_derivative, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(min, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              min,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(max, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              max,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(clamp, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        float,              min,
        float,              max,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(min, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(max, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(clamp, float, data, size,
        uint32_t,           size,
        const buf<float> &, data,
        const buf<float> &, min,
        const buf<float> &, max,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(compare, float, first, size,
        uint32_t,           size,
        const buf<float> &, first,
        const buf<float> &, second,
        bool *,             result)

DECLARE_MATH_FUNCTION(matvec_r, float, matrix, width * height,
        uint32_t,           width,
        uint32_t,           height,
        const buf<float> &, matrix,
        const buf<float> &, vector,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(matvec_c, float, matrix, width * height,
        uint32_t,           width,
        uint32_t,           height,
        const buf<float> &, matrix,
        const buf<float> &, vector,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(transpose, float, matrix, width * height,
        uint32_t,           width,
        uint32_t,           height,
        const buf<float> &, matrix,
        buf<float> &,       result)

DECLARE_MATH_FUNCTION(matmul_rc, float, cmatrix, width * height,
        uint32_t,           width,
        uint32_t,           height,
        uint32_t,           width2,
        const buf<float> &, rmatrix,
        const buf<float> &, cmatrix,
        buf<float> &,       result)

} // namespace nn
