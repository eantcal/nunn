add_executable(som_demo som_demo.cc)
target_link_libraries(som_demo PRIVATE nunn)
target_include_directories(som_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
    ${EIGEN3_INCLUDE_DIR}
)
