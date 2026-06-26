cmake_minimum_required(VERSION 3.14)
project(ocr_test VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ocr_test WIN32
    ocr_test.cpp
    stdafx.cpp
    ocr_test.rc
)

target_include_directories(ocr_test PRIVATE
    ${nunn_SOURCE_DIR}/common/inc
    ${nunn_SOURCE_DIR}/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/mnist
)

target_compile_definitions(ocr_test PRIVATE
    _CRT_SECURE_NO_WARNINGS
    WIN32
    _WINDOWS
)

target_link_libraries(ocr_test PRIVATE nunn mnist Comctl32)

if(MSVC)
    target_compile_options(ocr_test PRIVATE /W3 /permissive-)
endif()

install(TARGETS ocr_test
    RUNTIME DESTINATION bin
    COMPONENT runtime
)
