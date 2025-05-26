; Full code explanation is in the simd_support.asm file, written in MASM
; assembly.

global _get_simd_support

section .text

AVX512_DQ        EQU 0x00010000
AVX              EQU 0x00000800
SSE3             EQU 0x00000001
SIMD_UNSUPPORTED EQU 0
SIMD_SSE3        EQU 1
SIMD_AVX         EQU 2
SIMD_AVX512      EQU 3

_get_simd_support:
        PUSH rbp
        MOV rbp, rsp

        PUSH rbx
        PUSH rcx
        PUSH rdx

        MOV eax, 7
        MOV ecx, 0

        CPUID

        TEST ebx, AVX512_DQ
        JNZ avx512_supported

        MOV eax, 1
        MOV ecx, 0

        CPUID

        TEST ecx, AVX
        JNZ avx_supported

        TEST ecx, SSE3
        JNZ sse3_supported

        MOV eax, SIMD_UNSUPPORTED
        JMP return

avx512_supported:
        MOV eax, SIMD_AVX512
        JMP return

avx_supported:
        MOV eax, SIMD_AVX
        JMP return

sse3_supported:
        MOV eax, SIMD_SSE3

return:
        POP rdx
        POP rcx
        POP rbx
        POP rbp
        RET
