cmake_minimum_required(VERSION 3.13)
project(NeuralNetwork)

if (WIN32 AND BUILD_CUDA_SUPPORT AND NOT MSVC)
        message(FATAL_ERROR "MSVC is required for compilation when compiling "
                            "with CUDA on windows.")
endif ()

option(BUILD_CUDA_SUPPORT  "Build with CUDA support" ON)
option(BUILD_MNIST_DATASET "Build MNIST dataset"     ON)
option(BUILD_DECODER       "Build decoder"           ON)
option(BUILD_TESTS         "Build tests"             ON)
option(BUILD_MOCK          "Build mock"              ON)

if (BUILD_CUDA_SUPPORT)
        if (NOT DEFINED ENV{CUDA_PATH})
                message(WARNING "Unable to find Cuda Toolkit! Please install the CUDA "
                        "Toolkit from https://developer.nvidia.com/cuda-toolkit")

                set(BUILD_CUDA_SUPPORT OFF)
        else ()
                message(STATUS "Found CUDA Toolkit: $ENV{CUDA_PATH}")
        endif()
endif ()

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_compile_definitions(NDEBUG)
endif ()

# ------------------------------------------------------------------------------
# FLAGS
# ------------------------------------------------------------------------------

# C++
set(CMAKE_CXX_STANDARD          23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------

set(LIBS_DIR        "${CMAKE_CURRENT_LIST_DIR}/libs")
set(GET_PROJECT_DIR "${LIBS_DIR}/GetProject")

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

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/neural-network/src")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/neural-network")

if (BUILD_MNIST_DATASET)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/mnist")
endif ()

if (BUILD_DECODER)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/decoder")
endif ()
