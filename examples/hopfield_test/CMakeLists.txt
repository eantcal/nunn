cmake_minimum_required(VERSION 2.8.12)

include_directories(${nunn_SOURCE_DIR})
link_directories(${nunn_BINARY_DIR})
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
add_executable(hopfield_test ${SOURCES})

target_link_libraries(hopfield_test nunn)
