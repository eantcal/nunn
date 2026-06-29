cmake_minimum_required(VERSION 3.12)
project(linear_regression_demo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${nunn_SOURCE_DIR}/neural_networks/inc)
include_directories(${nunn_SOURCE_DIR}/common/inc)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
add_executable(linear_regression_demo ${SOURCES})
target_link_libraries(linear_regression_demo nunn)

if(MSVC)
    target_compile_options(linear_regression_demo PRIVATE /W4 /permissive-)
else()
    target_compile_options(linear_regression_demo PRIVATE -Wall -Wextra -pedantic)
endif()
