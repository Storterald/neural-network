cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)

# If the Network uses CUDA or not.
# This option should be disabled with small networks.
option(COMPILE_WITH_CUDA "Compile with CUDA" OFF)

# Targets
set(MNIST mnist)

# Directories
set(SRC_DIR ${CMAKE_HOME_DIRECTORY}/src)
set(MNIST_DIR ${CMAKE_HOME_DIRECTORY}/mnist)

# Required for the CONSTEXPR_SWITCH macro
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")

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
                ${SRC_DIR}/*.cpp
                ${SRC_DIR}/utils/*.cpp
                ${SRC_DIR}/cuda/*.cu
        )
else ()
        message(STATUS "Compiling without CUDA sources.")
        file(GLOB_RECURSE SOURCES
                ${SRC_DIR}/*.cpp
        )
endif ()

# Executables
add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(${MNIST} ${MNIST_DIR}/main.cpp)

if (COMPILE_WITH_CUDA)
        # Let CUDA compile only CUDA files
        set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Include headers and link static library
        target_include_directories(${PROJECT_NAME} PRIVATE $ENV{CUDA_PATH}/Include)
        target_link_libraries(${PROJECT_NAME} PRIVATE $ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib)
endif ()

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE BASE_PATH="${CMAKE_HOME_DIRECTORY}")
target_compile_definitions(${MNIST} PRIVATE BASE_PATH="${MNIST_DIR}")