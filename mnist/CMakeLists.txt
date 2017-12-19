cmake_minimum_required(VERSION 2.8.12)
project(mnist)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.)
include_directories(${nunn_SOURCE_DIR})

link_directories(${nunn_BINARY_DIR})

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(mnist SHARED ${SOURCES})
set_target_properties(mnist PROPERTIES SOVERSION ${VERSION_MAJOR}.${VERSION_MINOR} VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

target_link_libraries(mnist nunn)
