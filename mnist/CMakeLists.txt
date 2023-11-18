cmake_minimum_required(VERSION 3.12)
project(mnist VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Target-specific include directories
include_directories(${nunn_SOURCE_DIR}/common/inc)
include_directories(${nunn_SOURCE_DIR}/neural_networks/inc)

# File GLOB
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# Add library
add_library(mnist SHARED ${SOURCES})

# Set target properties
set_target_properties(mnist PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

# Link libraries
target_link_libraries(mnist nunn)

# Compiler-specific configurations
if(MSVC)
    # Using Visual Studio C++
    target_compile_options(mnist PRIVATE /W4 /permissive-)
else()
    # Other compilers like GCC, Clang
    target_compile_options(mnist PRIVATE -Wall -Wextra -pedantic)
endif()
