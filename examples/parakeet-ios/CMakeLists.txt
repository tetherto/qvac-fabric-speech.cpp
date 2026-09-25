cmake_minimum_required(VERSION 3.25)

if (NOT CMAKE_GENERATOR STREQUAL "Xcode")
    message(FATAL_ERROR "ParakeetLab must be generated with CMake's Xcode generator")
endif()

project(ParakeetLab LANGUAGES C CXX OBJC OBJCXX Swift)

set(CMAKE_XCODE_GENERATE_SCHEME ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PARAKEET_IOS_GGUF "" CACHE FILEPATH "Q8_0 GGUF copied into ParakeetLab.app")
set(PARAKEET_IOS_COREML "" CACHE PATH "Compiled encoder .mlmodelc copied into ParakeetLab.app")
set(PARAKEET_IOS_SAMPLE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../engines/parakeet/test/samples/jfk.wav"
    CACHE FILEPATH "Sample audio copied into ParakeetLab.app")

if (NOT EXISTS "${PARAKEET_IOS_GGUF}")
    message(FATAL_ERROR "Pass -DPARAKEET_IOS_GGUF=/absolute/path/to/parakeet-tdt-0.6b-v3.q8_0.gguf")
endif()
if (NOT IS_DIRECTORY "${PARAKEET_IOS_COREML}")
    message(FATAL_ERROR "Pass -DPARAKEET_IOS_COREML=/absolute/path/to/parakeet-tdt-0.6b-v3-encoder.mlmodelc")
endif()
if (NOT EXISTS "${PARAKEET_IOS_SAMPLE}")
    message(FATAL_ERROR "Sample audio not found: ${PARAKEET_IOS_SAMPLE}")
endif()

set(PARAKEET_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PARAKEET_INSTALL OFF CACHE BOOL "" FORCE)
set(PARAKEET_COREML ON CACHE BOOL "" FORCE)
set(PARAKEET_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_METAL_USE_BF16 ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../engines/parakeet"
    "${CMAKE_CURRENT_BINARY_DIR}/parakeet-native"
)

set(APP_SOURCES
    App/ParakeetLabApp.swift
    App/ContentView.swift
    App/BenchmarkViewModel.swift
    App/AudioPreparation.swift
    Native/ParakeetBridge.h
    Native/ParakeetBridge.mm
    App/ParakeetLab-Bridging-Header.h
)

add_executable(ParakeetLab MACOSX_BUNDLE ${APP_SOURCES})
target_link_libraries(ParakeetLab PRIVATE qvac-parakeet)
target_include_directories(ParakeetLab PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Native"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../engines/parakeet/include"
)

find_library(AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
target_link_libraries(ParakeetLab PRIVATE
    ${AVFOUNDATION_FRAMEWORK}
    ${COREML_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

set_target_properties(ParakeetLab PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/App/Info.plist"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "to.tether.qvac.ParakeetLab"
    XCODE_ATTRIBUTE_PRODUCT_NAME "Parakeet Lab"
    XCODE_ATTRIBUTE_MARKETING_VERSION "1.0"
    XCODE_ATTRIBUTE_CURRENT_PROJECT_VERSION "1"
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.4"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/App/ParakeetLab-Bridging-Header.h"
    XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++17"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
    XCODE_ATTRIBUTE_GENERATE_INFOPLIST_FILE "NO"
)

# The engine derives the sidecar path by removing `.q8_0.gguf` and appending
# `-encoder.mlmodelc`, so both resources must keep these exact bundle names.
add_custom_command(TARGET ParakeetLab POST_BUILD
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed-models.sh"
        "${PARAKEET_IOS_GGUF}"
        "${PARAKEET_IOS_COREML}"
        "${PARAKEET_IOS_SAMPLE}"
    COMMENT "Embedding Parakeet Q8_0 GGUF and compiled Core ML encoder"
    VERBATIM
)
