# Unit tests for the nunn library.
#
# GoogleTest is fetched at configure time (no submodule / system install needed).
# The first configure requires network access to clone googletest.

cmake_minimum_required(VERSION 3.14)
project(nunn_tests LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)

# On Windows, force GoogleTest to use the shared CRT (/MD, /MDd) so it matches
# the rest of the project and avoids LNK2038 runtime-mismatch errors.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# All test translation units in this directory.
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_executable(nunn_tests ${TEST_SOURCES})

target_include_directories(nunn_tests PRIVATE
    ${nunn_SOURCE_DIR}/common/inc
    ${nunn_SOURCE_DIR}/neural_networks/inc
    ${nunn_SOURCE_DIR}/reinforcement/inc
)

target_link_libraries(nunn_tests PRIVATE nunn GTest::gtest_main)

set_target_properties(nunn_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

# Register every TEST()/TEST_F() with CTest so `ctest` can run them.
include(GoogleTest)
gtest_discover_tests(nunn_tests)
