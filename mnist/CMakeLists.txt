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

function(mnist_configure_library target_name nunn_target)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        # On Windows a SHARED library only produces an import .lib if it exports
        # symbols; this example library uses no __declspec(dllexport), so export
        # everything automatically and let mnist_test link against it.
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    target_link_libraries(${target_name} ${nunn_target})

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -pedantic)
    endif()
endfunction()

# Add library
add_library(mnist STATIC ${SOURCES})
mnist_configure_library(mnist nunn)

if(NUNN_OCR_RUNTIME_FALLBACK)
    add_library(mnist_cpu STATIC ${SOURCES})
    mnist_configure_library(mnist_cpu nunn_cpu)
endif()
