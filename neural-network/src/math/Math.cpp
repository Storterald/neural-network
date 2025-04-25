#include "Math.h"

#include "../intrinsic/Intrinsic.h"
#include "_Math.h"

template<typename T>
static constexpr auto _get(T &&v, Data::DataLocation location)
{
        if constexpr (std::same_as<std::remove_cvref_t<T>, Data>)
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
        case SIMD_SSE:                                                                          \
                return _Math<MATH_SSE>:: __name__ (GET_ALL(Data::HOST, __VA_ARGS__));           \
        case SIMD_UNSUPPORTED:                                                                  \
                return _Math<MATH_NORMAL>:: __name__ (GET_ALL(Data::HOST, __VA_ARGS__));        \
        }                                                                                       \
} while (false)

void Math::sum(uint32_t size, const Data &first, const Data &second, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(sum, size, size, first, second, result);
        SIMD_SWITCH(sum, size, first, second, result);
}

void Math::sub(uint32_t size, const Data &first, const Data &second, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(sub, size, size, first, second, result);
        SIMD_SWITCH(sub, size, first, second, result);
}

void Math::mul(uint32_t size, const Data &first, const Data &second, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(mul, size, size, first, second, result);
        SIMD_SWITCH(mul, size, first, second, result);
}

void Math::div(uint32_t size, const Data &first, const Data &second, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(div, size, size, first, second, result);
        SIMD_SWITCH(div, size, first, second, result);
}

void Math::sum(uint32_t size, const Data &data, float scalar, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(sum, size, size, data, scalar, result);
        SIMD_SWITCH(sum, size, data, scalar, result);
}

void Math::sub(uint32_t size, const Data &data, float scalar, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(sub, size, size, data, scalar, result);
        SIMD_SWITCH(sub, size, data, scalar, result);
}

void Math::mul(uint32_t size, const Data &data, float scalar, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(mul, size, size, data, scalar, result);
        SIMD_SWITCH(mul, size, data, scalar, result);
}

void Math::div(uint32_t size, const Data &data, float scalar, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(div, size, size, data, scalar, result);
        SIMD_SWITCH(div, size, data, scalar, result);
}

void Math::tanh(uint32_t size, const Data &data, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(tanh, size, size, data, result);
        SIMD_SWITCH(tanh, size, data, result);
}

void Math::tanh_derivative(uint32_t size, const Data &data, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(tanh_derivative, size, size, data, result);
        SIMD_SWITCH(tanh_derivative, size, data, result);
}

void Math::ReLU(uint32_t size, const Data &data, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(ReLU, size, size, data, result);
        SIMD_SWITCH(ReLU, size, data, result);
}

void Math::ReLU_derivative(uint32_t size, const Data &data, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(ReLU_derivative, size, size, data, result);
        SIMD_SWITCH(ReLU_derivative, size, data, result);
}

void Math::min(uint32_t size, const Data &data, float min, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(min, size, size, data, min, result);
        SIMD_SWITCH(min, size, data, min, result);
}

void Math::max(uint32_t size, const Data &data, float max, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(max, size, size, data, max, result);
        SIMD_SWITCH(max, size, data, max, result);
}

void Math::clamp(uint32_t size, const Data &data, float min, float max, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(clamp, size, size, data, min, max, result);
        SIMD_SWITCH(clamp, size, data, min, max, result);
}

void Math::min(uint32_t size, const Data &first, const Data &second, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(min, size, size, first, second, result);
        SIMD_SWITCH(min, size, first, second, result);
}

void Math::max(uint32_t size, const Data &first, const Data &second, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(max, size, size, first, second, result);
        SIMD_SWITCH(max, size, first, second, result);
}

void Math::clamp(uint32_t size, const Data &data, const Data &min, const Data &max, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(clamp, size, size, data, min, max, result);
        SIMD_SWITCH(clamp, size, data, min, max, result);
}

void Math::matvec_mul(uint32_t width, uint32_t height, const Data &matrix, const Data &vector, Data &result)
{
        CHECK_IF_CUDA_MINIMUM(matvec_mul, width * height, width, height, matrix, vector, result);
        SIMD_SWITCH(matvec_mul, width, height, matrix, vector, result);
}