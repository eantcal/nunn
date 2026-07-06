cmake_minimum_required(VERSION 3.14)
project(ocr_test VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(_MODEL_SEARCH_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/scripts/mnist/models"
)

function(add_ocr_app target_name nunn_target mnist_target)
    add_executable(${target_name} WIN32
        ocr_test.cpp
        stdafx.cpp
        ocr_test.rc
    )

    target_include_directories(${target_name} PRIVATE
        ${nunn_SOURCE_DIR}/common/inc
        ${nunn_SOURCE_DIR}/neural_networks/inc
        ${CMAKE_SOURCE_DIR}/mnist
    )

    target_compile_definitions(${target_name} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32
        _WINDOWS
    )

    target_link_libraries(${target_name} PRIVATE ${nunn_target} ${mnist_target} Comctl32 Shell32 Advapi32)

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W3 /permissive-)
    endif()

    foreach(_dir ${_MODEL_SEARCH_DIRS})
        file(GLOB _json_files "${_dir}/*.json")
        foreach(_f ${_json_files})
            get_filename_component(_fname ${_f} NAME)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:${target_name}>/models"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_f}"
                    "$<TARGET_FILE_DIR:${target_name}>/models/${_fname}"
                COMMENT "Deploying model ${_fname}"
                VERBATIM
            )
        endforeach()
    endforeach()
endfunction()

if(NUNN_OCR_RUNTIME_FALLBACK)
    add_executable(ocr_test WIN32
        ocr_launcher.cpp
        ocr_test.rc
    )
    target_compile_definitions(ocr_test PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32
        _WINDOWS
    )
    target_link_libraries(ocr_test PRIVATE Shell32)

    if(MSVC)
        target_compile_options(ocr_test PRIVATE /W3 /permissive-)
    endif()

    add_ocr_app(ocr_test_opencl nunn mnist)
    nunn_deploy_arrayfire_runtime(ocr_test_opencl)

    add_ocr_app(ocr_test_cpu nunn_cpu mnist_cpu)
    add_dependencies(ocr_test ocr_test_opencl ocr_test_cpu)

    install(TARGETS ocr_test ocr_test_opencl ocr_test_cpu
        RUNTIME DESTINATION bin
        COMPONENT runtime
    )
else()
    add_ocr_app(ocr_test nunn mnist)
    nunn_deploy_arrayfire_runtime(ocr_test)

    install(TARGETS ocr_test
        RUNTIME DESTINATION bin
        COMPONENT runtime
    )
endif()
