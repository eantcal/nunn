cmake_minimum_required(VERSION 3.12)
project(nunn VERSION 2.1.0 LANGUAGES CXX)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/.
    ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/neural_networks/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/reinforcement/inc
)

# File GLOB
file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/common/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/neural_networks/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/reinforcement/src/*.cc"
)

function(nunn_configure_library target_name)
    target_link_libraries(${target_name} PRIVATE nlohmann_json::nlohmann_json)
    # Eigen is header-only; PUBLIC so that consumers that include nu_mlpmatrixnn.h
    # can find <Eigen/Core> without an extra target_link_libraries call.
    target_link_libraries(${target_name} PUBLIC Eigen3::Eigen)

    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    )

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -pedantic)
    endif()
endfunction()

# Add library
add_library(nunn STATIC ${SOURCES})
nunn_configure_library(nunn)

# ArrayFire — optional GPU/OpenCL backend for MlpMatrixNN
if(NUNN_ENABLE_OPENCL AND ArrayFire_OpenCL_FOUND)
    target_link_libraries(nunn PUBLIC ArrayFire::afopencl)
    target_compile_definitions(nunn PUBLIC NUNN_HAS_ARRAYFIRE)
endif()

if(NUNN_OCR_RUNTIME_FALLBACK)
    add_library(nunn_cpu STATIC ${SOURCES})
    nunn_configure_library(nunn_cpu)
endif()

