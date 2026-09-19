# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.22)

# Need to switch back to gcc to custom use SIMD instructions
# error: instruction requires the following: 'Espressif ESPV 2.1'
# set(IDF_TOOLCHAIN clang CACHE STRING "Toolchain")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(super_potato)

add_subdirectory(scripts)
