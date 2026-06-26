cmake_minimum_required(VERSION 3.14)
project(net2json LANGUAGES CXX)

add_executable(net2json net2json.cc)

target_include_directories(net2json PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/reinforcement/inc
)

target_link_libraries(net2json PRIVATE nunn)

set_target_properties(net2json PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)
