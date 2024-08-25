#pragma once

#ifndef NDEBUG
#define DEBUG_MODE_ENABLED
#endif

#include <cstdint>

constexpr float EPSILON { 1e-8 };
constexpr float LEARNING_RATE { 5e-5f };
constexpr uint32_t SIMD_WIDTH { 16U };
constexpr uint32_t MAX_FILE_SIZE { (uint32_t)1e9f };
constexpr uint32_t BLOCK_BITSHIFT { 8 };
constexpr uint32_t BLOCK_SIZE { 1 << BLOCK_BITSHIFT };

// CUDA cannot compile with Zc:preprocessor
// This stops the compilation here only when compiling CUDA files
#ifndef __CUDACC__

// Allows a macro to expand 342 times. Credit:
// https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define PARENS ()
#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

// Helper macro, creates a code block with everything but the first parameter.
// already contains a ';' at the end of the vararg to make a single line case
// look better, if the codeblock contains multiple lines, they will need to be
// separated using a ';'.
#define CASE(value, ...) value, { __VA_ARGS__; }

// Creates a constexpr if else chain inside a do { ... } while(false) block
// to allow the macro to be inside a block without {}. Also obligates the user
// to put ';' after it keeping a cleaner and more uniform syntax.
#define CONSTEXPR_SWITCH(var, ...)                                              \
        do {                                                                    \
                __VA_OPT__(EXPAND(__CONSTEXPR_SWITCH(var, __VA_ARGS__)))        \
        } while (false)

// The actual macro definition.
#define __CONSTEXPR_SWITCH(var, case1, codeBlock1, ...)                         \
        if constexpr (var == case1)                                             \
                codeBlock1                                                      \
        __VA_OPT__(else __CONSTEXPR_SWITCH2 PARENS (var, __VA_ARGS__))

#define __CONSTEXPR_SWITCH2() __CONSTEXPR_SWITCH

#endif // __CUDACC__