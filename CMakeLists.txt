cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES CXX CUDA)

# Flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")

# If the Network uses CUDA or not.
# This option should be disabled with small networks.
option(COMPILE_WITH_CUDA "Compile with CUDA" OFF)

# Targets
set(MNIST mnist)
set(DECODER decoder)

# ----------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------

set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/src")
set(DECODER_DIR "${CMAKE_HOME_DIRECTORY}/decoder")
set(LIBS_DIR "${CMAKE_HOME_DIRECTORY}/libs")
set(CMAKE_UTILS_DIR "${LIBS_DIR}/CMakeUtils")
set(OPENCV_DIR "${LIBS_DIR}/opencv")
set(MNIST_DIR "${CMAKE_HOME_DIRECTORY}/mnist")

# Network sources
if (COMPILE_WITH_CUDA)
        message(STATUS "Compiling with CUDA sources.")

        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)

        # Check if CUDA toolkit is installed
        find_package(CUDAToolkit REQUIRED)

        # Set CMake CUDA related flags
        set(CMAKE_CUDA_COMPILER $ENV{CUDA_PATH}/bin/nvcc)
        set(CUDA_INCLUDE_DIRS $ENV{CUDA_PATH}/Include)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

        # Tell the files to use CUDA
        add_compile_definitions(USE_CUDA)

        # Don't include math/*.cpp
        file(GLOB SOURCES
                "${SRC_DIR}/*.cpp"
                "${SRC_DIR}/utils/*.cpp"
                "${SRC_DIR}/cuda/*.cu"
        )
else ()
        message(STATUS "Compiling without CUDA sources.")
        file(GLOB_RECURSE SOURCES
                "${SRC_DIR}/*.cpp"
        )
endif ()

# ----------------------------------------------------------------------------------------------------------------------
# PROJECT
# ----------------------------------------------------------------------------------------------------------------------

# Network
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE BASE_PATH="${CMAKE_HOME_DIRECTORY}")

# Mnist dataset encoder
add_executable(${MNIST} ${MNIST_DIR}/main.cpp)
target_compile_definitions(${MNIST} PRIVATE BASE_PATH="${MNIST_DIR}")

# Encoded file visualiser
add_executable(${DECODER} ${DECODER_DIR}/main.cpp)
target_compile_definitions(${DECODER} PRIVATE BASE_PATH="${CMAKE_HOME_DIRECTORY}")

# ----------------------------------------------------------------------------------------------------------------------
# TARGET OPTIONS
# ----------------------------------------------------------------------------------------------------------------------

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_definitions(${PROJECT_NAME} PUBLIC DEBUG_MODE_ENABLED)
endif ()

if (COMPILE_WITH_CUDA)
        # Let CUDA compile only CUDA files
        set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Include headers and link static library
        target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/Include")
        target_link_libraries(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib")
endif ()

# ----------------------------------------------------------------------------------------------------------------------
# DEPENDENCIES
# ----------------------------------------------------------------------------------------------------------------------

# Fetch CMake Utils
if (NOT EXISTS ${CMAKE_UTILS_DIR})
        execute_process(
                COMMAND git clone "https://github.com/Storterald/CMake-Utils" --branch "main" --single-branch ${CMAKE_UTILS_DIR}
        )
else ()
        execute_process(
                COMMAND git pull "https://github.com/Storterald/CMake-Utils"
                WORKING_DIRECTORY ${CMAKE_UTILS_DIR}
        )
endif ()
include("${CMAKE_UTILS_DIR}/get_library.cmake")