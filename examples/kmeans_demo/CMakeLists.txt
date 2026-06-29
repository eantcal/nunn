add_executable(kmeans_demo kmeans_demo.cc)
target_link_libraries(kmeans_demo PRIVATE nunn)
target_include_directories(kmeans_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/nunn/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/nunn/common/inc
)
