#include "SIMD.h"

SIMD getSIMDSupport() {
        int eax = 0, ebx = 0, ecx = 0, edx = 0;

        _cpuid(7, 0, &eax, &ebx, &ecx, &edx);
        if (ebx & (1 << 16))
                return AVX512;

        _cpuid(1, 0, &eax, &ebx, &ecx, &edx);
        if (ecx & (1 << 28))
                return AVX;

        if (edx & (1 << 25))
                return SSE;

        return UNSUPPORTED;
}