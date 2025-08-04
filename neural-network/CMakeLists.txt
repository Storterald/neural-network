cmake_minimum_required(VERSION 3.13)
project(NeuralNetworkLib LANGUAGES CXX)

# ------------------------------------------------------------------------------
# FLAGS
# ------------------------------------------------------------------------------

# C++
if (MSVC)
        # Silenced warnings:
        #   C4464 relative path in include
        #   C4514 unused inline function removed
        #   C4548 comma expression has no effect
        #   C4625 copy constructor implicitly deleted
        #   C4626 copy assignment op implicitly deleted
        #   C4702 unreachable code
        #   C4710 function not inlined
        #   C4711 function inlined
        #   C4820 implicit padding
        #   C5026 move constructor implicitly deleted
        #   C5027 move assignment op implicitly deleted
        #   C5045 spectre mitigation
        set(CMAKE_CXX_FLAGS         "/Zc:preprocessor /EHa /openmp /Wall /WX /wd4464 /wd4514 /wd4548 /wd4625 /wd4626 /wd4702 /wd4710 /wd4711 /wd4820 /wd5026 /wd5027 /wd5045")
        set(CMAKE_CXX_FLAGS_DEBUG   "/MTd /Zi /Od")
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /MT")

        if (NOT IS_X86_64)
                # Silence volatile access of '<expression>' is subject to setting
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4746")
        endif ()
else ()
        set(CMAKE_CXX_FLAGS         "-Wall -Wextra -Wpedantic -Werror -fopenmp")
        set(CMAKE_CXX_FLAGS_DEBUG   "-gdwarf-4")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -static")

        if (IS_X86_64)
                if ("${SIMD_SUPPORT_LEVEL}" GREATER_EQUAL "1")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
                endif ()

                if ("${SIMD_SUPPORT_LEVEL}" GREATER_EQUAL "2")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")
                endif ()

                if ("${SIMD_SUPPORT_LEVEL}" GREATER_EQUAL "3")
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq")
                endif ()
        endif ()
endif ()

# ASM x86
if (IS_X86_64)
        set(CMAKE_ASM_MASM_FLAGS_INIT  "/c /nologo")
        set(CMAKE_ASM_MASM_FLAGS_DEBUG "/Zi")
        set(CMAKE_ASM_NASM_FLAGS_INIT  "")
        set(CMAKE_ASM_NASM_FLAGS_DEBUG "-g")
endif ()

# Cuda
if (BUILD_CUDA_SUPPORT)
        set(CMAKE_CUDA_COMPILER               "$ENV{CUDA_PATH}/bin/nvcc.exe")
        set(CMAKE_CUDA_STANDARD               20)
        set(CMAKE_CUDA_STANDARD_REQUIRED      ON)
        set(CMAKE_CUDA_ARCHITECTURES          80 89 120)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)
        set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

        set(CMAKE_CUDA_FLAGS         "-diag-suppress 221")
        set(CMAKE_CUDA_FLAGS_DEBUG   "")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

        if (MSVC)
                # Silences the same warnings as C++ plus:
                #   C4100 unused parameter
                #   C4505 unreferenced function with internal linkage
                #   C4619 no warning number <n>
                #   C4668 not defined as preprocessor macro
                #   C5031 #pragma warning(pop) mismatch
                #   C5039 C++ exception in extern "C"
                set(CMAKE_CUDA_FLAGS         "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor,/EHs,/Wall,/WX,/wd4100,/wd4464,/wd4505,/wd4514,/wd4548,/wd4619,/wd4625,/wd4626,/wd4668,/wd4702,/wd4710,/wd4711,/wd4820,/wd5026,/wd5027,/wd5031,/wd5039,/wd5045")
                set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS} -Xcompiler=/MTd,/Zi,/Od")
                set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -Xcompiler=/MT")
        endif ()
endif ()

# ------------------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------------------

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

add_library(NeuralNetworkHeaders INTERFACE)
target_include_directories(NeuralNetworkHeaders INTERFACE ${INCLUDE_DIR})
add_library(NeuralNetwork::Headers ALIAS NeuralNetworkHeaders)

file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp")

add_library(NeuralNetworkLib STATIC ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(NeuralNetworkLib PRIVATE NeuralNetwork::Headers)
add_library(NeuralNetwork::Lib ALIAS NeuralNetworkLib)

if (BUILD_CUDA_SUPPORT)
        enable_language(CUDA)
        target_compile_definitions(NeuralNetworkLib
                PUBLIC BUILD_CUDA_SUPPORT)

        add_library(CudaRT STATIC IMPORTED)
        set_property(TARGET CudaRT PROPERTY IMPORTED_LOCATION "$ENV{CUDA_PATH}/Lib/x64/cudart_static.lib")
        add_library(Cuda::RTLib ALIAS CudaRT)

        file(GLOB_RECURSE CUDA_SOURCES
                "${SRC_DIR}/*.cu")

        target_sources(NeuralNetworkLib
                PUBLIC ${CUDA_SOURCES})
        target_include_directories(NeuralNetworkLib
                PUBLIC "$ENV{CUDA_PATH}/Include")
        target_link_libraries(NeuralNetworkLib
                PUBLIC Cuda::RTLib)
endif ()

# ------------------------------------------------------------------------------
# SUB DIRECTORIES
# ------------------------------------------------------------------------------

if (BUILD_TESTS)
        cmake_policy(SET CMP0079 NEW)  # Added in CMake 3.13

        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
        add_dependencies(NeuralNetworkTest NeuralNetwork::Lib)

        target_link_libraries(NeuralNetworkTest
                PRIVATE NeuralNetwork::Lib
                PRIVATE NeuralNetwork::Headers)
endif ()

if (BUILD_MOCK)
        cmake_policy(SET CMP0079 NEW)

        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/mock")
        add_dependencies(NeuralNetworkMock NeuralNetwork::Lib)

        target_link_libraries(NeuralNetworkMock
                PRIVATE NeuralNetwork::Lib
                PRIVATE NeuralNetwork::Headers)
endif ()
