cmake_minimum_required(VERSION 3.12)
project(tictactoe VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Target-specific include directories
include_directories(${nunn_SOURCE_DIR}/common/inc)
include_directories(${nunn_SOURCE_DIR}/neural_networks/inc)

# File GLOB
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# Add executable
add_executable(tictactoe ${SOURCES})

# Link libraries
target_link_libraries(tictactoe nunn)

# Compiler-specific configurations
if(MSVC)
    # Using Visual Studio C++
    target_compile_options(tictactoe PRIVATE /W4 /permissive-)
else()
    # Other compilers like GCC, Clang
    target_compile_options(tictactoe PRIVATE -Wall -Wextra -pedantic)
endif()

# Set target properties (optional, for more fine-grained control)
set_target_properties(tictactoe PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

