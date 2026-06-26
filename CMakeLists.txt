cmake_minimum_required(VERSION 3.14)
project(nunnlib)

# Set binary and runtime output directories
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Configuring build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
    message("Setting build type to '${CMAKE_BUILD_TYPE}'")
else()
    message("Build type set to '${CMAKE_BUILD_TYPE}'")
endif()

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific configurations
if(MSVC)
    # Using Visual Studio C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    endif()
else()
    # Other compilers (like GCC, Clang)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# nlohmann/json — fetched before subdirectories that use it
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# Add subdirectories
add_subdirectory(nunn)
add_subdirectory(mnist)
add_subdirectory(nunn_topo)
add_subdirectory(examples)
add_subdirectory(tools)

# Unit tests (GoogleTest, fetched via FetchContent)
option(NUNN_BUILD_TESTS "Build the nunn unit tests" ON)
if(NUNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install ───────────────────────────────────────────────────────────────────

# Demo executables
install(TARGETS
        and_test counter_test hopfield_test maze path_finder
        tictactoe xor_test titanic nunn_topo mnist_test
    RUNTIME DESTINATION bin
    COMPONENT runtime
)

if(WIN32)
    install(TARGETS ocr_test winttt
        RUNTIME DESTINATION bin
        COMPONENT runtime
    )
endif()

# mnist helper DLL (runtime) / import lib (devel)
install(TARGETS mnist
    RUNTIME DESTINATION bin   # .dll on Windows
    LIBRARY DESTINATION lib   # .so on Linux/macOS
    ARCHIVE DESTINATION lib   # import .lib on Windows
    COMPONENT runtime
)

# net2json conversion tool
install(TARGETS net2json
    RUNTIME DESTINATION bin
    COMPONENT devel
)

# nunn static library
install(TARGETS nunn
    ARCHIVE DESTINATION lib
    COMPONENT devel
)

# Public headers
foreach(_dir common/inc neural_networks/inc reinforcement/inc)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/nunn/${_dir}/
        DESTINATION include/nunn
        COMPONENT devel
        FILES_MATCHING PATTERN "*.h"
    )
endforeach()

# Pre-trained JSON model files
install(FILES
        ${CMAKE_SOURCE_DIR}/examples/ocr_test/nn_125hl.json
        ${CMAKE_SOURCE_DIR}/examples/tictactoe/tictactoe.json
    DESTINATION share/nunn/nets
    COMPONENT nets
)

# Documentation
install(FILES
        ${CMAKE_SOURCE_DIR}/README.md
        ${CMAKE_SOURCE_DIR}/NEWS
        ${CMAKE_SOURCE_DIR}/COPYING
    DESTINATION share/nunn/doc
    COMPONENT docs
)

# ── CPack ─────────────────────────────────────────────────────────────────────
set(CPACK_PACKAGE_NAME                  "nunn")
set(CPACK_PACKAGE_VENDOR                "Antonino Calderone")
set(CPACK_PACKAGE_CONTACT               "antonino.calderone@gmail.com")
set(CPACK_PACKAGE_VERSION               "2.1.0")
set(CPACK_PACKAGE_VERSION_MAJOR         "2")
set(CPACK_PACKAGE_VERSION_MINOR         "1")
set(CPACK_PACKAGE_VERSION_PATCH         "0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY   "Nunn - C++20 machine learning library")
set(CPACK_RESOURCE_FILE_LICENSE         "${CMAKE_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README          "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_INSTALL_DIRECTORY     "Nunn ${CPACK_PACKAGE_VERSION}")

# Components
set(CPACK_COMPONENTS_ALL runtime devel nets docs)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Demo applications")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION  "Console demo programs (xor_test, tictactoe, mnist_test, ...)")
set(CPACK_COMPONENT_RUNTIME_REQUIRED     TRUE)
set(CPACK_COMPONENT_DEVEL_DISPLAY_NAME   "Development files")
set(CPACK_COMPONENT_DEVEL_DESCRIPTION    "Static library and headers for building against nunn")
set(CPACK_COMPONENT_NETS_DISPLAY_NAME    "Pre-trained models")
set(CPACK_COMPONENT_NETS_DESCRIPTION     "JSON model files for ocr_test and tictactoe")
set(CPACK_COMPONENT_NETS_SELECTED        TRUE)
set(CPACK_COMPONENT_DOCS_DISPLAY_NAME    "Documentation")
set(CPACK_COMPONENT_DOCS_DESCRIPTION     "README, NEWS, LICENSE")
set(CPACK_COMPONENT_DOCS_SELECTED        TRUE)

# NSIS (exe installer, requires NSIS installed)
set(CPACK_NSIS_PACKAGE_NAME             "Nunn ${CPACK_PACKAGE_VERSION}")
set(CPACK_NSIS_DISPLAY_NAME             "Nunn Library ${CPACK_PACKAGE_VERSION}")
set(CPACK_NSIS_INSTALL_ROOT             "$PROGRAMFILES64")
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
set(CPACK_NSIS_MODIFY_PATH              OFF)
set(CPACK_NSIS_MUI_ICON                 "")
set(CPACK_NSIS_MUI_UNIICON              "")

# WiX (MSI installer, requires WiX Toolset v3/v4)
# Reuse the GUID from the old Inno Setup scripts.
set(CPACK_WIX_UPGRADE_GUID              "FAB3A99E-879A-4177-8DB0-2BCFB23E48DB")
set(CPACK_WIX_PRODUCT_NAME              "Nunn Library ${CPACK_PACKAGE_VERSION}")

# Default generator on Windows: NSIS if available, otherwise ZIP
if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
endif()

include(CPack)
