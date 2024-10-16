#pragma once

// Avoiding inline ASM code in C++ with the __asm__ block. Using ASM file directly.
extern void _cpuid(int ID, int subID, int *eax, int *ebx, int *ecx, int *edx);

typedef enum SIMD {
        UNSUPPORTED,
        SSE,
        AVX,
        AVX512
} SIMD;

SIMD getSIMDSupport();