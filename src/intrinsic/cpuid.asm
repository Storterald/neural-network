.CODE
PUBLIC _cpuid

; Function _cpuid(int id, int subID, int* eax, int* ebx, int* ecx, int* edx)
_cpuid PROC
        ; Save address of previous stack frame
        PUSH rbp
        ; Address of current stack frame
        MOV rbp, rsp

        ; Load ID and Sub ID in eax and ecx respectly
        MOV eax, [rbp + 16]
        MOV ecx, [rbp + 24]

        CPUID

        MOV [rbp + 32], eax
        MOV [rbp + 40], ebx
        MOV [rbp + 48], ecx
        MOV [rbp + 56], edx

        ; Restore stack pointer
        POP rbp
        RET

_cpuid ENDP
END