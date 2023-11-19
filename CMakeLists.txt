cmake_minimum_required(VERSION 3.12)
project(nunnlib)

# Set binary and runtime output directories
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Configuring build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
    message("Setting build type to '${CMAKE_BUILD_TYPE}'")
else()
    message("Build type set to '${CMAKE_BUILD_TYPE}'")
endif()

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific configurations
if(MSVC)
    # Using Visual Studio C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    endif()
else()
    # Other compilers (like GCC, Clang)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Add subdirectories
add_subdirectory(nunn)
add_subdirectory(mnist)
add_subdirectory(nunn_topo)
add_subdirectory(examples)
