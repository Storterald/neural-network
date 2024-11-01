.CODE
PUBLIC getSIMDSupport

SIMD_UNSUPPORTED EQU 0
SIMD_SSE EQU 1
SIMD_AVX EQU 2
SIMD_AVX512 EQU 3

; SIMD getSIMDSupport()
getSIMDSupport PROC
        PUSH rbp
        MOV rbp, rsp

        ; Flags for CPUID
        MOV eax, 7
        MOV ecx, 0

        ; https://en.wikipedia.org/wiki/CPUID
        CPUID

        ; TEST performs a bitwise AND (&).
        TEST ebx, 00010000H ; 1 << 16
        JNZ AVX512_SUPPORTED

        TEST ecx, 10000000H ; 1 << 26
        JNZ AVX_SUPPORTED

        TEST edx, 02000000H ; 1 << 25
        JNZ SSE_SUPPORTED

        MOV eax, SIMD_UNSUPPORTED
        JMP END_SIMD

AVX512_SUPPORTED:
        MOV eax, SIMD_AVX512
        JMP END_SIMD

AVX_SUPPORTED:
        MOV eax, SIMD_AVX
        JMP END_SIMD

SSE_SUPPORTED:
        MOV eax, SIMD_SSE
        JMP END_SIMD

END_SIMD:
        POP rbp
        RET

getSIMDSupport ENDP
END