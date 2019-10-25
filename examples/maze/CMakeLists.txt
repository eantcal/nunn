cmake_minimum_required(VERSION 2.8.12)

include_directories(${nunn_SOURCE_DIR}/reinforcement/inc)
include_directories(${nunn_SOURCE_DIR}/common/inc)

link_directories(${nunn_BINARY_DIR})

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_executable(maze ${SOURCES})

target_link_libraries(maze nunn)
