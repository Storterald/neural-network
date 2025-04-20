cmake_minimum_required(VERSION 3.13)
project(NeuralNetwork LANGUAGES CXX ASM_MASM CUDA)

if (NOT MSVC)
        message(FATAL_ERROR "MSVC is required for compilation.")
endif ()

if (NOT DEFINED ENV{CUDA_PATH})
        message(FATAL_ERROR "Unable to find Cuda Toolkit! Please install the CUDA "
                            "Toolkit from https://developer.nvidia.com/cuda-toolkit")
else()
        message(STATUS "Found CUDA Toolkit: $ENV{CUDA_PATH}")
endif()

option(BUILD_MNIST_DATASET     "Build MNIST dataset"     ON)
option(BUILD_DECODER           "Build decoder"           ON)
option(BUILD_TESTS             "Build tests"             ON)
option(BUILD_MOCK              "Build mock"              ON)

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------

set(SRC_DIR         "${CMAKE_CURRENT_LIST_DIR}/src")
set(LIBS_DIR        "${CMAKE_CURRENT_LIST_DIR}/libs")
set(GET_PROJECT_DIR "${LIBS_DIR}/GetProject")

# ------------------------------------------------------------------------------
# FLAGS
# ------------------------------------------------------------------------------

# C++
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS             "/Zc:preprocessor /EHs /W4 /wd4711 /wd4514 /wd4577 /wd4710 /wd4625 /wd4626 /wd5045 /wd5026 /wd5027 /wd4820 /wd4866")
set(CMAKE_CXX_FLAGS_DEBUG       "/MTd")
set(CMAKE_CXX_FLAGS_RELEASE     "/O2 /MT")

# ASM
set(CMAKE_ASM_MASM_FLAGS_INIT  "/c /nologo")
set(CMAKE_ASM_MASM_FLAGS_DEBUG "/Zi")

# Cuda
set(CMAKE_CUDA_COMPILER               "$ENV{CUDA_PATH}/bin/nvcc.exe")
set(CMAKE_CUDA_STANDARD               20)
set(CMAKE_CUDA_STANDARD_REQUIRED      ON)
set(CMAKE_CUDA_ARCHITECTURES          "native")
set(CMAKE_CUDA_SEPARABLE_COMPILATION  ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_CUDA_FLAGS_INIT             "-Xcompiler=/Zc:preprocessor")
set(CMAKE_CUDA_FLAGS_DEBUG            "-Xcompiler=/MTd")
set(CMAKE_CUDA_FLAGS_RELEASE          "-O3 -Xcompiler=/MT")

# ------------------------------------------------------------------------------
# Cuda Libraries
# ------------------------------------------------------------------------------

add_library(CudaRT STATIC IMPORTED)
set_property(TARGET CudaRT PROPERTY IMPORTED_LOCATION "$ENV{CUDA_PATH}/Lib/x64/cudart_static.lib")
add_library(Cuda::RTLib ALIAS CudaRT)

# ------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.asm"
        "${SRC_DIR}/*.cu")

add_library(NeuralNetworkLib STATIC ${SOURCES})
add_library(NeuralNetwork::Lib ALIAS NeuralNetworkLib)
target_include_directories(NeuralNetworkLib
        PUBLIC "$ENV{CUDA_PATH}/Include")
target_link_libraries(NeuralNetworkLib
        PUBLIC Cuda::RTLib)

# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------

if (NOT EXISTS ${GET_PROJECT_DIR})
        execute_process(
                COMMAND git clone "https://github.com/Storterald/GetProject.git"
                        --quiet --branch "main" --single-branch ${GET_PROJECT_DIR})
else ()
        execute_process(
                COMMAND git pull --quiet
                WORKING_DIRECTORY ${GET_PROJECT_DIR})
endif ()

include("${GET_PROJECT_DIR}/GetProject.cmake")

# ------------------------------------------------------------------------------
# SUB DIRECTORIES
# ------------------------------------------------------------------------------

if (BUILD_MNIST_DATASET)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/mnist")
        target_compile_definitions(Mnist
                PUBLIC BASE_PATH="${CMAKE_CURRENT_LIST_DIR}")
endif ()

if (BUILD_DECODER)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/decoder")
        target_compile_definitions(Decoder
                PUBLIC BASE_PATH="${CMAKE_CURRENT_LIST_DIR}")
endif ()

if (BUILD_TESTS)
        cmake_policy(SET CMP0079 NEW)  # Added in CMake 3.13
        
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
        add_dependencies(NeuralNetworkTest NeuralNetwork::Lib)

        target_link_libraries(NeuralNetworkTest
                PRIVATE NeuralNetwork::Lib)
        target_include_directories(NeuralNetworkTest
                PRIVATE ${SRC_DIR})
endif ()

if (BUILD_MOCK)
        cmake_policy(SET CMP0079 NEW)

        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/mock")
        add_dependencies(NeuralNetworkMock NeuralNetwork::Lib)

        target_link_libraries(NeuralNetworkMock
                PRIVATE NeuralNetwork::Lib)
        target_include_directories(NeuralNetworkMock
                PRIVATE ${SRC_DIR})
endif ()