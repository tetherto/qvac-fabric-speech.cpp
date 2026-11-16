# Standalone build of the shared Core ML runtime's portable tests, for working
# on engines/coreml without configuring an engine (no ggml needed):
#   cmake -S engines/coreml -B build && cmake --build build && ctest --test-dir build
# Engines fold the runtime into their own libraries via qvac_coreml.cmake.
cmake_minimum_required(VERSION 3.20)
project(qvac-coreml LANGUAGES C CXX)

if (APPLE)
    enable_language(OBJC OBJCXX)
endif()

enable_testing()
include(${CMAKE_CURRENT_SOURCE_DIR}/qvac_coreml.cmake)
qvac_coreml_add_tests()
