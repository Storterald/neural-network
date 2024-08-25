#pragma once

#include "../Base.h"

#ifdef USE_CUDA
#error "To include math/Base.h, it is required to compile without CUDA support."
#endif

#ifdef DEBUG_MODE_ENABLED
#include "../utils/Logger.h"
#endif // DEBUG_MODE_ENABLED

#if !defined DEBUG_MODE_ENABLED && !defined DISABLE_AVX512
#include <immintrin.h>
#endif