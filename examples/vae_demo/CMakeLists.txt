add_executable(vae_demo vae_demo.cc)
target_link_libraries(vae_demo PRIVATE nunn)
target_include_directories(vae_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
    ${EIGEN3_INCLUDE_DIR})
