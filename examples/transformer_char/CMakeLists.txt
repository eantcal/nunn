add_executable(transformer_char transformer_char.cc)
target_link_libraries(transformer_char PRIVATE nunn)
target_include_directories(transformer_char PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
    ${EIGEN3_INCLUDE_DIR}
)
