cmake_minimum_required(VERSION 2.8.12)
project(nunn)
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(nunn SHARED ${SOURCES})
set_target_properties(nunn PROPERTIES SOVERSION ${VERSION_MAJOR}.${VERSION_MINOR} VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

# target_link_libraries(nunn todo)
