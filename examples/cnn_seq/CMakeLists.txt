cmake_minimum_required(VERSION 3.12)
project(cnn_seq)

add_executable(cnn_seq cnn_seq.cc)

target_include_directories(cnn_seq PRIVATE
    ${nunn_SOURCE_DIR}/common/inc
    ${nunn_SOURCE_DIR}/neural_networks/inc
    ${nunn_SOURCE_DIR}/reinforcement/inc
)

target_link_libraries(cnn_seq PRIVATE nunn)

set_target_properties(cnn_seq PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)
