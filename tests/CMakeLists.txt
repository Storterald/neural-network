cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkTest LANGUAGES CXX CUDA)

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

# ------------------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------------------

# All tests sources
file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.cu")

add_executable(NeuralNetworkTest ${SOURCES})

target_compile_definitions(NeuralNetworkTest
        PRIVATE DEBUG_MODE_ENABLED)

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

target_include_directories(NeuralNetworkTest
        PRIVATE "${googletest_SOURCE}/googletest/include")
target_link_libraries(NeuralNetworkTest
        PRIVATE gtest
        PRIVATE gtest_main)