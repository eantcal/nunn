cmake_minimum_required(VERSION 3.14)
project(winttt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(winttt WIN32
    winttt.cc
    stdafx.cpp
    winttt.rc
)

target_include_directories(winttt PRIVATE
    ${nunn_SOURCE_DIR}/common/inc
    ${nunn_SOURCE_DIR}/neural_networks/inc
)

target_compile_definitions(winttt PRIVATE
    _CRT_SECURE_NO_WARNINGS
    WIN32
    _WINDOWS
)

target_link_libraries(winttt PRIVATE nunn Comctl32)

if(MSVC)
    target_compile_options(winttt PRIVATE /W3 /permissive-)

    # Precompiled header — replicates the /Yc / /Yu setup from the old .vcxproj
    target_compile_options(winttt PRIVATE /Yu"stdafx.h")
    set_source_files_properties(stdafx.cpp
        PROPERTIES COMPILE_FLAGS "/Yc\"stdafx.h\""
    )
endif()

install(TARGETS winttt
    RUNTIME DESTINATION bin
    COMPONENT runtime
)
