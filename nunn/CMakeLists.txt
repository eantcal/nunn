cmake_minimum_required(VERSION 3.12)
project(nunn VERSION 2.0.0 LANGUAGES CXX)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/.
    ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/neural_networks/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/reinforcement/inc
)

# File GLOB
file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/common/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/neural_networks/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/reinforcement/src/*.cc"
)

# Add library
add_library(nunn SHARED ${SOURCES})

# Set target properties
set_target_properties(nunn PROPERTIES 
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES 
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)

# Compiler-specific configurations
if(MSVC)
    # Using Visual Studio C++
    target_compile_options(nunn PRIVATE /W4 /permissive-)
else()
    # Other compilers like GCC, Clang
    target_compile_options(nunn PRIVATE -Wall -Wextra -pedantic)
endif()

