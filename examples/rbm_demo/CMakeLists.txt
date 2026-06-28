add_executable(rbm_demo rbm_demo.cc)
target_link_libraries(rbm_demo PRIVATE nunn)
target_include_directories(rbm_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
    ${EIGEN3_INCLUDE_DIR})
