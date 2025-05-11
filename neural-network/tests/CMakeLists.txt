cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkTest LANGUAGES CXX)

# ------------------------------------------------------------------------------
# FLAGS
# ------------------------------------------------------------------------------

# C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4189")

# ------------------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------------------

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")

file(GLOB_RECURSE SOURCES "${SRC_DIR}/*.cpp")
add_executable(NeuralNetworkTest ${SOURCES})

target_compile_definitions(NeuralNetworkTest
        PRIVATE DEBUG_MODE_ENABLED)

if (BUILD_CUDA_SUPPORT)
        enable_language(CUDA)

        file(GLOB_RECURSE CUDA_SOURCES "${SRC_DIR}/*.cu")
        target_sources(NeuralNetworkTest PRIVATE ${CUDA_SOURCES})
endif ()

# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------

get_project(
        GIT_REPOSITORY "https://github.com/google/googletest.git"
        VERSION LATEST
        OPTIONS
                BUILD_GMOCK=OFF
                gtest_force_shared_crt=ON
                INSTALL_GTEST=OFF)

if (NOT googletest_ADDED)
        message(FATAL_ERROR "Could not get googletest.")
endif ()

target_include_directories(NeuralNetworkTest
        PRIVATE "${googletest_SOURCE}/googletest/include")
target_link_libraries(NeuralNetworkTest
        PRIVATE gtest
        PRIVATE gtest_main)
