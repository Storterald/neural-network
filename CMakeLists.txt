cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES C CXX ASM_MASM CUDA)

# ----------------------------------------------------------------------------------------------------------------------
# REQUIREMENTS
# ----------------------------------------------------------------------------------------------------------------------

if (NOT MSVC)
        message(FATAL_ERROR "MSVC is required for compilation.")
endif ()

# Check if CUDA toolkit is installed
find_package(CUDAToolkit REQUIRED)

# ----------------------------------------------------------------------------------------------------------------------
# FLAGS
# ----------------------------------------------------------------------------------------------------------------------

# Compiler Flags
set(CMAKE_MSVC_RUNTIME_LIBRARY "")
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "")

# Linker Flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

# ASM Flags
set(CMAKE_ASM_MASM_FLAGS_RELEASE  "/c /nologo")
set(CMAKE_ASM_MASM_FLAGS_DEBUG  "/c /nologo /Zi")

# C Flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MDd")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MD")

# C++ Flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")

# CUDA Flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER $ENV{CUDA_PATH}/bin/nvcc)
set(CUDA_INCLUDE_DIRS $ENV{CUDA_PATH}/Include)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# ----------------------------------------------------------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------

set(SRC_DIR "${CMAKE_HOME_DIRECTORY}/src")
set(DECODER_DIR "${CMAKE_HOME_DIRECTORY}/decoder")
set(CMAKE_UTILS_DIR "${LIBS_DIR}/CMakeUtils")
set(OPENCV_DIR "${LIBS_DIR}/opencv")
set(MNIST_DIR "${CMAKE_HOME_DIRECTORY}/mnist")

# ----------------------------------------------------------------------------------------------------------------------
# PROJECT
# ----------------------------------------------------------------------------------------------------------------------

# The network sources
file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.c"
        "${SRC_DIR}/*.asm"
        "${SRC_DIR}/*.cu"
)

# Network
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE BASE_PATH="${CMAKE_HOME_DIRECTORY}")

# Mnist dataset encoder
set(MNIST mnist)
add_executable(${MNIST} "${MNIST_DIR}/main.cpp")
target_compile_definitions(${MNIST} PRIVATE BASE_PATH="${MNIST_DIR}")

# Encoded file visualiser
set(DECODER decoder)
add_executable(${DECODER} "${DECODER_DIR}/main.cpp")
target_compile_definitions(${DECODER} PRIVATE BASE_PATH="${CMAKE_HOME_DIRECTORY}")

# ----------------------------------------------------------------------------------------------------------------------
# TARGET OPTIONS
# ----------------------------------------------------------------------------------------------------------------------

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_definitions(${PROJECT_NAME} PUBLIC DEBUG_MODE_ENABLED)
endif ()

# Let CUDA compile only CUDA files
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include headers and link static library
target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/Include")
target_link_libraries(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib")