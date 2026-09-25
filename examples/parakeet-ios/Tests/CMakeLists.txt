cmake_minimum_required(VERSION 3.25)
project(ParakeetIOSLongAudioIntegration LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PARAKEET_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PARAKEET_INSTALL OFF CACHE BOOL "" FORCE)
set(PARAKEET_COREML ON CACHE BOOL "" FORCE)
set(PARAKEET_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/parakeet"
    "${CMAKE_CURRENT_BINARY_DIR}/parakeet-native"
)

add_executable(parakeet-ios-long-audio-integration
    long_audio_integration.cpp
    ../Native/LongAudioRunner.cpp
)
target_link_libraries(parakeet-ios-long-audio-integration PRIVATE qvac-parakeet)
target_include_directories(parakeet-ios-long-audio-integration PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../Native"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/parakeet/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../engines/parakeet/src"
)
