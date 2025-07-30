function (get_simd_support OUTPUT_VARIABLE)
        if (MSVC)
                enable_language(ASM_MASM)
                set(ASM_EXT "asm")
        else ()
                enable_language(ASM_NASM)
                set(CMAKE_ASM_NASM_COMPILER "nasm")
                set(ASM_EXT "nasm")
        endif ()

        try_run(_ COMPILATION_RESULT
                "${CMAKE_BINARY_DIR}/simd_support"
                SOURCES "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/simd_support.c"
                        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/simd_support.${ASM_EXT}"
                CMAKE_FLAGS CMAKE_ASM_NASM_COMPILER=${CMAKE_ASM_NASM_COMPILER}
                RUN_OUTPUT_VARIABLE LEVEL)

        set(${OUTPUT_VARIABLE} ${LEVEL} PARENT_SCOPE)
endfunction ()