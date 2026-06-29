add_executable(pca_demo pca_demo.cc)
target_link_libraries(pca_demo PRIVATE nunn)
target_include_directories(pca_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
    ${EIGEN3_INCLUDE_DIR}
)
