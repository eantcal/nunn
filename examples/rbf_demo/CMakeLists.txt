cmake_minimum_required(VERSION 3.12)
project(rbf_demo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${nunn_SOURCE_DIR}/neural_networks/inc)
include_directories(${nunn_SOURCE_DIR}/common/inc)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
add_executable(rbf_demo ${SOURCES})
target_link_libraries(rbf_demo nunn)

set_target_properties(rbf_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(rbf_demo PRIVATE /W4 /permissive-)
else()
    target_compile_options(rbf_demo PRIVATE -Wall -Wextra -pedantic)
endif()
