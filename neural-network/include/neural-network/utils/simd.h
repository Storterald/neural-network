#pragma once

#ifndef TARGET_X86_64
#error utils/simd.h cannot be included if the architecture is not x86_64
#endif // !TARGET_X86_64

#if SIMD_SUPPORT_LEVEL   == 1
#include "_simd128.h"
#elif SIMD_SUPPORT_LEVEL == 2
#include "_simd256.h"
#elif SIMD_SUPPORT_LEVEL == 3
#include "_simd512.h"
#endif // SIMD_SUPPORT_LEVEL

namespace nn::simd {

#if SIMD_SUPPORT_LEVEL   == 0
using simd = void;
#elif SIMD_SUPPORT_LEVEL == 1
using simd = _m128;
using mask = _mask4;
#elif SIMD_SUPPORT_LEVEL == 2
using simd = _m256;
using mask = _mask8;
#elif SIMD_SUPPORT_LEVEL == 3
using simd = _m512;
using mask = _mask16;
#endif // SIMD_SUPPORT_LEVEL

} // namespace nn::simd
