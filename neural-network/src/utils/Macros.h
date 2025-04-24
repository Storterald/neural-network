#pragma once

// Allows a macro to expand 342 times. Credit:
// https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define PARENS ()
#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

#define GET_ARGS_NAMES(...) __VA_OPT__(EXPAND(__GET_ARGS_NAMES(__VA_ARGS__)))

#define __GET_ARGS_NAMES(type, name, ...)                               \
        name __VA_OPT__(, __GET_ARGS_NAMES2 PARENS (__VA_ARGS__))

#define __GET_ARGS_NAMES2() __GET_ARGS_NAMES

#define GET_ARGS(...) __VA_OPT__(EXPAND(__GET_ARGS(__VA_ARGS__)))

#define __GET_ARGS(type, name, ...)                               \
        type name __VA_OPT__(, __GET_ARGS2 PARENS (__VA_ARGS__))

#define __GET_ARGS2() __GET_ARGS