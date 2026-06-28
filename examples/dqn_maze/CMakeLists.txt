cmake_minimum_required(VERSION 3.12)
project(dqn_maze)

add_executable(dqn_maze dqn_maze.cc)

target_include_directories(dqn_maze PRIVATE
    ${nunn_SOURCE_DIR}/common/inc
    ${nunn_SOURCE_DIR}/neural_networks/inc
    ${nunn_SOURCE_DIR}/reinforcement/inc
)

target_link_libraries(dqn_maze PRIVATE nunn)

set_target_properties(dqn_maze PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)
