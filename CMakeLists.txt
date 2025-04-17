cmake_minimum_required(VERSION 3.28)
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
option(BUILD_AS_STATIC_LIBRARY "Build as static library" OFF)

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------

set(SRC_DIR         "${CMAKE_HOME_DIRECTORY}/src")
set(LIBS_DIR        "${CMAKE_HOME_DIRECTORY}/libs")
set(GET_PROJECT_DIR "${LIBS_DIR}/GetProject")

# ------------------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.asm"
        "${SRC_DIR}/*.cu")

if (NOT BUILD_AS_STATIC_LIBRARY)
        add_executable(NeuralNetwork ${SOURCES})
        add_executable(NeuralNetwork::Exe ALIAS NeuralNetwork)

        target_compile_definitions(NeuralNetwork
                PRIVATE BASE_PATH="${CMAKE_HOME_DIR}")
        target_include_directories(NeuralNetwork
                PRIVATE "$ENV{CUDA_PATH}/Include")
        target_link_libraries(NeuralNetwork
                PRIVATE "$ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib")
endif ()

if (BUILD_AS_STATIC_LIBRARY OR BUILD_TESTS)
        list(REMOVE_ITEM SOURCES "${SRC_DIR}/main.cpp")
        add_library(NeuralNetworkLib STATIC ${SOURCES})
        add_library(NeuralNetwork::Lib ALIAS NeuralNetworkLib)

        target_compile_definitions(NeuralNetworkLib
                PUBLIC BASE_PATH="${CMAKE_SOURCE_DIR}")
        target_include_directories(NeuralNetworkLib
                PUBLIC "$ENV{CUDA_PATH}/Include")
        target_link_libraries(NeuralNetworkLib
                PRIVATE "$ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib")
endif ()

# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------

# Fetch the latest commit of GetProject
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
                PUBLIC BASE_PATH="${CMAKE_HOME_DIR}")
endif ()

if (BUILD_DECODER)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/decoder")
        target_compile_definitions(Decoder
                PUBLIC BASE_PATH="${CMAKE_HOME_DIR}")
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