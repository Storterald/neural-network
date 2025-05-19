cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkMock LANGUAGES CXX)

# ------------------------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------------------------

add_executable(NeuralNetworkMock "main.cpp")

# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------

get_project(
        URL "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h"
        LIBRARY_NAME "stb"
        FILE ON)

if (NOT stb_DOWNLOADED)
        message(FATAL_ERROR "Could not get stb image.")
endif ()

target_include_directories(NeuralNetworkMock
        PRIVATE "${stb_SOURCE}")