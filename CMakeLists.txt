cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES C CXX ASM_MASM CUDA)

# ------------------------------------------------------------------------------
# REQUIREMENTS
# ------------------------------------------------------------------------------

if (NOT MSVC)
        message(FATAL_ERROR "MSVC is required for compilation.")
endif ()

if (NOT DEFINED ENV{CUDA_PATH})
        message(FATAL_ERROR "Unable to find Cuda Toolkit! Please install the CUDA "
                "Toolkit from https://developer.nvidia.com/cuda-toolkit")
else()
        message(STATUS "Found CUDA Toolkit: $ENV{CUDA_PATH}")
endif()

# ------------------------------------------------------------------------------
# FLAGS
# ------------------------------------------------------------------------------

# Compiler Flags
set(CMAKE_MSVC_RUNTIME_LIBRARY "")
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "")

# Linker Flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

# ASM Flags
set(CMAKE_ASM_MASM_FLAGS_RELEASE "/c /nologo")
set(CMAKE_ASM_MASM_FLAGS_DEBUG "/c /nologo /Zi")

# C++ Flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")

# CUDA Flags
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe")
set(CUDA_INCLUDE_DIRS "$ENV{CUDA_PATH}/Include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/MDd")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xcompiler=/MD")

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(MNIST_DIR "${CMAKE_SOURCE_DIR}/mnist")
set(DECODER_DIR "${CMAKE_SOURCE_DIR}/decoder")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
set(LIBS_DIR "${CMAKE_SOURCE_DIR}/libs")
set(CMAKE_UTILS_DIR "${LIBS_DIR}/CMakeUtils")
set(GTEST_DIR "${LIBS_DIR}/googletest")

# ------------------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------------------

# All network sources
file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.c"
        "${SRC_DIR}/*.asm"
        "${SRC_DIR}/*.cu"
)

# Network
add_executable(${PROJECT_NAME} ${SOURCES})

# Mnist dataset encoder
set(MNIST mnist)
add_executable(${MNIST} "${MNIST_DIR}/main.cpp")

# Encoded file visualiser
set(DECODER decoder)
add_executable(${DECODER} "${DECODER_DIR}/main.cpp")

# Exclude main.cpp from compiling in tests and static lib
list(REMOVE_ITEM SOURCES "${SRC_DIR}/main.cpp")

# Network static lib
set(NETWORK_LIB "${PROJECT_NAME}Lib")
add_library(${NETWORK_LIB} STATIC ${SOURCES})

# All tests sources
file(GLOB_RECURSE TESTS_SOURCES
        "${TESTS_DIR}/*.cpp"
        "${TESTS_DIR}/*.cu"
)

# Tests
set(TESTS tests)
add_executable(${TESTS} ${SOURCES} ${TESTS_SOURCES})

# ------------------------------------------------------------------------------
# TARGETS OPTIONS
# ------------------------------------------------------------------------------

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        add_compile_definitions(DEBUG_MODE_ENABLED)
else ()
        # Tests should always be run with debug features enabled.
        target_compile_definitions(${TESTS} PRIVATE DEBUG_MODE_ENABLED)
endif ()

# For easier use the path of the source folder is passed to the code.
add_compile_definitions(BASE_PATH="${CMAKE_SOURCE_DIR}")

# Disable stdout logging during tests.
target_compile_definitions(${TESTS} PRIVATE LOGGER_FATAL_ERROR_DISABLE_STDOUT)

# Let CUDA compile only CUDA files
set_target_properties(${PROJECT_NAME} ${TESTS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include headers and link static library
target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/Include")
target_link_libraries(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib")
target_include_directories(${TESTS} PRIVATE "$ENV{CUDA_PATH}/Include")
target_link_libraries(${TESTS} PRIVATE "$ENV{CUDA_PATH}/Lib/x64/cudadevrt.lib")

# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------

# Fetch CMake Utils
if (NOT EXISTS ${CMAKE_UTILS_DIR})
        execute_process(
                COMMAND git clone "https://github.com/Storterald/CMake-Utils"
                        --branch "main" --single-branch ${CMAKE_UTILS_DIR}
        )
else ()
        execute_process(
                COMMAND git pull "https://github.com/Storterald/CMake-Utils"
                WORKING_DIRECTORY ${CMAKE_UTILS_DIR}
        )
endif ()

include("${CMAKE_UTILS_DIR}/get_library.cmake")

# Fetch GLFW
build_latest_release(
        TARGET ${TESTS}
        PROFILE_NAME "google"
        REPOSITORY_NAME "googletest"
        DIRECTORY ${LIBS_DIR}
        INSTALL_ENABLED ON
        BUILD_ARGS
                -DBUILD_GMOCK=OFF
                -Dgtest_force_shared_crt=ON
)
target_include_directories(${TESTS} PRIVATE "${GTEST_DIR}/include")
target_link_libraries(${TESTS} PRIVATE "${GTEST_DIR}/lib/gtest.lib")
target_link_libraries(${TESTS} PRIVATE "${GTEST_DIR}/lib/gtest_main.lib")