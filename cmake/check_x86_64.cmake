function (check_x86_64 OUTPUT_VARIABLE)
        try_compile(IS_X86_64
                "${CMAKE_BINARY_DIR}"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_x86_64.c")

        if (NOT IS_X86_64)
                set(${OUTPUT_VARIABLE} OFF PARENT_SCOPE)
        else ()
                set(${OUTPUT_VARIABLE} ON PARENT_SCOPE)
        endif ()
endfunction ()