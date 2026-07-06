cmake_minimum_required(VERSION 3.14)
project(ocr_test VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ocr_test WIN32
    ocr_test.cpp
    stdafx.cpp
    ocr_test.rc
)

target_include_directories(ocr_test PRIVATE
    ${nunn_SOURCE_DIR}/common/inc
    ${nunn_SOURCE_DIR}/neural_networks/inc
    ${CMAKE_SOURCE_DIR}/mnist
)

target_compile_definitions(ocr_test PRIVATE
    _CRT_SECURE_NO_WARNINGS
    WIN32
    _WINDOWS
)

target_link_libraries(ocr_test PRIVATE nunn mnist Comctl32 Shell32 Advapi32)
nunn_deploy_arrayfire_runtime(ocr_test)

if(MSVC)
    target_compile_options(ocr_test PRIVATE /W3 /permissive-)
endif()

# Copy pre-trained model JSON files into a models/ subdir next to the exe
set(_MODEL_SEARCH_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/scripts/mnist/models"
)
foreach(_dir ${_MODEL_SEARCH_DIRS})
    file(GLOB _json_files "${_dir}/*.json")
    foreach(_f ${_json_files})
        get_filename_component(_fname ${_f} NAME)
        add_custom_command(TARGET ocr_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:ocr_test>/models"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_f}"
                "$<TARGET_FILE_DIR:ocr_test>/models/${_fname}"
            COMMENT "Deploying model ${_fname}"
            VERBATIM
        )
    endforeach()
endforeach()

install(TARGETS ocr_test
    RUNTIME DESTINATION bin
    COMPONENT runtime
)
