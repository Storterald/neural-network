cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 23)

if (MSVC)
        # Required for the CONSTEXPR_SWITCH macro
        add_compile_options("/Zc:preprocessor")
else ()
        # Required to compile AVX512 stuff
        add_compile_options("-march=native")
endif ()

# Targets
set(MNIST mnist)
set(DECODER decoder)

# Directories
set(SRC_DIR ${CMAKE_HOME_DIRECTORY}/src)
set(MNIST_DIR ${CMAKE_HOME_DIRECTORY}/mnist)
set(DECODER_DIR ${CMAKE_HOME_DIRECTORY}/decoder)

# Network sources
file(GLOB_RECURSE SOURCES
        ${SRC_DIR}/*.cpp
)

# Network executable
add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(${MNIST} ${MNIST_DIR}/main.cpp)
add_executable(${DECODER} ${DECODER_DIR}/main.cpp)

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE BASE_PATH="${CMAKE_HOME_DIRECTORY}")
target_compile_definitions(${MNIST} PRIVATE BASE_PATH="${MNIST_DIR}")
target_compile_definitions(${DECODER} PRIVATE BASE_PATH="${DECODER_DIR}")