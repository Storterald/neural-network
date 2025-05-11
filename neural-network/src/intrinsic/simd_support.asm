.CODE
PUBLIC _get_simd_support

; This is the AVX512DQ instruction set bit, included in all currently existing
; CPUs with AVX512F.
AVX512_DQ        EQU 00010000H
; This is not the AVX instruction set bit (08000000H), instead the FMA bit, which
; means AVX support and fused multiply-add support. This will classify the listed
; CPU generations as SSE3 ones, as they only support the AVX instruction set:
;   - Pentium
;   - Celeron
;   - Sandy Bridge
;   - Ivy Bridge
;   - Bulldozer (without Piledriver cores)
AVX              EQU 00000800H
SSE3             EQU 00000001H

; simd enum
SIMD_UNSUPPORTED EQU 0
SIMD_SSE3        EQU 1
SIMD_AVX         EQU 2
SIMD_AVX512      EQU 3

; simd _get_simd_support()
_get_simd_support PROC
        PUSH rbp
        MOV rbp, rsp

        ; Preserve the register overwritten by CPUID
        PUSH rbx
        PUSH rcx
        PUSH rdx

        ; eax = 7, ecx = 0 : Extended Features
        MOV eax, 7
        MOV ecx, 0

        ; https://en.wikipedia.org/wiki/CPUID
        CPUID

        TEST ebx, AVX512_DQ
        JNZ avx512_supported

        ; eax = 1, ecx = 0 : Processor Info and Feature Bits
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

_get_simd_support ENDP
END
