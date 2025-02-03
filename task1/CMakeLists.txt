cmake_minimum_required(VERSION 3.10)
project(task1)

set(TYPE "float" CACHE STRING "")

if(TYPE STREQUAL "double")
    add_compile_definitions(USE_DOUBLE)
endif()

add_executable(task1 task1.cpp)