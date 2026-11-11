# Shared Core ML runtime (PLAN §2.3). Folded into each engine library under its
# own namespace; no new library is installed and the headers under include/
# are BUILD_INTERFACE only.
#
#   qvac_coreml_add_to(<engine-lib> NAMESPACE <ns> PREFIX <P> ENABLE <bool> [TEMPLATES <dirs>])
#       Compiles the runtime into <engine-lib> with QVAC_COREML_NS=<ns> when
#       ENABLE is true (the engine's Apple-only <P>_COREML option). <P> is the
#       engine's CMake option prefix (TTS_CPP, PARAKEET, AUDIOGEN). Template
#       embedding (TEMPLATES) arrives with S2b and is rejected until then.
#   qvac_coreml_add_tests()
#       Portable tests (Linux and macOS): test-coreml-scheduler,
#       test-coreml-recipe, test-coreml-milblob, test-coreml-cache, plus the
#       host CLI qvac-coreml-tool when a `ggml` target exists (the generator's
#       GgufSource and the tool need ggml-base; the tests then also cover real
#       GGUFs). Each engine calls it from its *_BUILD_TESTS block; the first call
#       wins in the umbrella build.
#   qvac_coreml_add_link_test(TARGETS <libs...>)
#       macOS umbrella only: test-coreml-link links every listed engine library
#       and all of their runtime copies into one binary.
include_guard(GLOBAL)

function(_qvac_coreml_runtime_sources out_var with_objc)
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(srcs
        "${dir}/src/hints.cpp"
        "${dir}/src/manager.cpp"
        "${dir}/src/options.cpp"
        "${dir}/src/pin.cpp"
        "${dir}/src/scheduler.cpp"
        "${dir}/src/sha256.cpp"
        "${dir}/src/status.cpp")
    if (with_objc)
        list(APPEND srcs "${dir}/src/apple/device.mm")
    else()
        list(APPEND srcs "${dir}/src/device_posix.cpp")
    endif()
    set(${out_var} ${srcs} PARENT_SCOPE)
endfunction()

# The portable generator (S2a, PLAN §2.6-§2.8): recipe parser and interpreter,
# MILBlob writer, keys, cache and failure records. No ggml. gguf_source.cpp
# (ggml-base) is separate. Compiled with -ffp-contract=off: the interpreter's
# arithmetic must be one IEEE operation per op on every host (§2.6).
function(_qvac_coreml_generator_sources out_var)
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(${out_var}
        "${dir}/src/cache.cpp"
        "${dir}/src/digest.cpp"
        "${dir}/src/failed.cpp"
        "${dir}/src/interp.cpp"
        "${dir}/src/milblob.cpp"
        "${dir}/src/recipe.cpp"
        "${dir}/src/source_stat.cpp"
        PARENT_SCOPE)
endfunction()

# Frameworks the Apple device facts need (src/apple/device.mm).
function(_qvac_coreml_apple_frameworks out_var)
    find_library(QVAC_COREML_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    set(fws "${QVAC_COREML_FOUNDATION_FRAMEWORK}")
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # SecCodeCopySelf (host identity for hints) is macOS-only.
        find_library(QVAC_COREML_SECURITY_FRAMEWORK Security REQUIRED)
        list(APPEND fws "${QVAC_COREML_SECURITY_FRAMEWORK}")
    endif()
    set(${out_var} ${fws} PARENT_SCOPE)
endfunction()

function(_qvac_coreml_configure_runtime target ns with_objc)
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    target_compile_definitions(${target} PRIVATE QVAC_COREML_NS=${ns})
    target_include_directories(${target} PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party")
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET     hidden
        OBJCXX_VISIBILITY_PRESET  hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if (with_objc)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    endif()
endfunction()

function(qvac_coreml_add_to target)
    cmake_parse_arguments(QCA "" "NAMESPACE;PREFIX;ENABLE" "TEMPLATES" ${ARGN})
    if (NOT TARGET ${target})
        message(FATAL_ERROR "qvac_coreml_add_to: ${target} is not a target")
    endif()
    if (NOT QCA_NAMESPACE OR NOT QCA_PREFIX)
        message(FATAL_ERROR "qvac_coreml_add_to(${target}): NAMESPACE and PREFIX are required")
    endif()
    if (QCA_TEMPLATES)
        message(FATAL_ERROR "qvac_coreml_add_to(${target}): template embedding (TEMPLATES) is not available yet (S2b)")
    endif()
    if (NOT QCA_ENABLE)
        return()  # no Core ML in this build: nothing compiled, no behaviour change
    endif()
    if (NOT APPLE)
        message(FATAL_ERROR "qvac_coreml_add_to(${target}): the Core ML runtime is Apple-only")
    endif()

    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(rt ${target}.qvac-coreml)
    _qvac_coreml_runtime_sources(srcs ON)
    add_library(${rt} OBJECT ${srcs})
    _qvac_coreml_configure_runtime(${rt} ${QCA_NAMESPACE} ON)
    target_sources(${target} PRIVATE $<TARGET_OBJECTS:${rt}>)

    # Engine sources that adopt the runtime (S1b, P1, A1) see the same
    # namespace and headers; nothing in the engine uses them yet.
    target_compile_definitions(${target} PRIVATE QVAC_COREML_NS=${QCA_NAMESPACE})
    target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${dir}/include>)

    # PUBLIC like each engine's own Core ML frameworks, for static consumers.
    _qvac_coreml_apple_frameworks(fws)
    target_link_libraries(${target} PUBLIC ${fws})
    set(pc_libs "")
    foreach (fw IN LISTS fws)
        get_filename_component(fw_name "${fw}" NAME_WE)
        list(APPEND pc_libs "-framework ${fw_name}")
    endforeach()

    set_property(TARGET ${target} PROPERTY QVAC_COREML_NS ${QCA_NAMESPACE})
    set_property(TARGET ${target} PROPERTY QVAC_COREML_PREFIX ${QCA_PREFIX})
    set_property(TARGET ${target} PROPERTY QVAC_COREML_RUNTIME ${rt})
    # For engines that write a pkg-config Libs.private (qvac-parakeet.pc).
    set_property(TARGET ${target} PROPERTY QVAC_COREML_PKGCONFIG_LIBS "${pc_libs}")
    message(STATUS "${target}: Core ML runtime compiled in (namespace ${QCA_NAMESPACE})")
endfunction()

function(qvac_coreml_add_tests)
    if (WIN32 OR TARGET test-coreml-scheduler)
        return()  # Linux and macOS only; registered once per build tree
    endif()
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    find_package(Threads REQUIRED)

    set(with_objc OFF)
    if (APPLE AND CMAKE_OBJCXX_COMPILER_LOADED)
        set(with_objc ON)
    endif()
    set(fws "")
    if (with_objc)
        _qvac_coreml_apple_frameworks(fws)
    endif()

    # The tests' own copy of the runtime and the generator, in its own namespace.
    _qvac_coreml_runtime_sources(srcs ${with_objc})
    _qvac_coreml_generator_sources(gen_srcs)
    set(have_ggml OFF)
    if (TARGET ggml)
        set(have_ggml ON)
        list(APPEND gen_srcs "${dir}/src/gguf_source.cpp")
    endif()
    add_library(qvac-coreml-test-runtime STATIC ${srcs} ${gen_srcs})
    _qvac_coreml_configure_runtime(qvac-coreml-test-runtime qvac_coreml_test ${with_objc})
    # Scheduler test seams (src/test_hooks.h): only in this copy.
    target_compile_definitions(qvac-coreml-test-runtime PRIVATE QVAC_COREML_TEST_HOOKS=1)
    target_compile_options(qvac-coreml-test-runtime PRIVATE -ffp-contract=off)
    target_link_libraries(qvac-coreml-test-runtime PUBLIC Threads::Threads ${CMAKE_DL_LIBS} ${fws})
    if (have_ggml)
        target_link_libraries(qvac-coreml-test-runtime PUBLIC ggml)
        target_compile_definitions(qvac-coreml-test-runtime PUBLIC QVAC_COREML_HAVE_GGML=1)
    endif()

    # A dlopen-able module with a third copy, for the dlclose lifetime case.
    add_library(qvac-coreml-test-plugin MODULE "${dir}/test/test_plugin.cpp" ${srcs})
    _qvac_coreml_configure_runtime(qvac-coreml-test-plugin qvac_coreml_plugin ${with_objc})
    target_link_libraries(qvac-coreml-test-plugin PRIVATE Threads::Threads ${CMAKE_DL_LIBS} ${fws})

    add_executable(test-coreml-scheduler "${dir}/test/test_scheduler.cpp")
    target_compile_definitions(test-coreml-scheduler PRIVATE QVAC_COREML_NS=qvac_coreml_test)
    target_include_directories(test-coreml-scheduler PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party")
    target_compile_features(test-coreml-scheduler PRIVATE cxx_std_17)
    target_link_libraries(test-coreml-scheduler PRIVATE qvac-coreml-test-runtime)
    add_dependencies(test-coreml-scheduler qvac-coreml-test-plugin)

    add_test(NAME test-coreml-scheduler
             COMMAND test-coreml-scheduler --plugin $<TARGET_FILE:qvac-coreml-test-plugin>)
    # Own label so CI runs it in a dedicated step (no SKIP path exists: any
    # non-zero exit, including 77, is a failure).
    set_tests_properties(test-coreml-scheduler PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 300)

    # The portable generator's tests (S2a): per-test temp dirs, no network, no
    # Core ML. Same label and no-skip rule as the scheduler test.
    foreach (t recipe milblob cache)
        add_executable(test-coreml-${t} "${dir}/test/test_${t}.cpp")
        target_compile_definitions(test-coreml-${t} PRIVATE QVAC_COREML_NS=qvac_coreml_test)
        target_include_directories(test-coreml-${t} PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party" "${dir}/test")
        target_compile_features(test-coreml-${t} PRIVATE cxx_std_17)
        target_compile_options(test-coreml-${t} PRIVATE -ffp-contract=off)
        target_link_libraries(test-coreml-${t} PRIVATE qvac-coreml-test-runtime)
        set(args "")
        if (t STREQUAL "recipe" AND have_ggml)
            set(args --tool $<TARGET_FILE:qvac-coreml-tool>)
        endif()
        add_test(NAME test-coreml-${t} COMMAND test-coreml-${t} ${args})
        set_tests_properties(test-coreml-${t} PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 300)
    endforeach()

    if (have_ggml)
        # Host CLI (PLAN §2.3): dump | twin | f32 | generate | digest | verify.
        add_executable(qvac-coreml-tool "${dir}/tools/qvac-coreml-tool.cpp")
        target_compile_definitions(qvac-coreml-tool PRIVATE QVAC_COREML_NS=qvac_coreml_test)
        target_include_directories(qvac-coreml-tool PRIVATE "${dir}/include" "${dir}/src" "${dir}/third_party")
        target_compile_features(qvac-coreml-tool PRIVATE cxx_std_17)
        target_link_libraries(qvac-coreml-tool PRIVATE qvac-coreml-test-runtime)
        # test-coreml-recipe drives `verify` through the CLI as well.
        add_dependencies(test-coreml-recipe qvac-coreml-tool)
    else()
        message(STATUS "qvac-coreml-tool and the GGUF cases of the generator tests: not built (no ggml target)")
    endif()
endfunction()

function(qvac_coreml_add_link_test)
    cmake_parse_arguments(QLT "" "" "TARGETS" ${ARGN})
    if (NOT APPLE OR TARGET test-coreml-link)
        return()
    endif()
    set(dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(gen "${CMAKE_CURRENT_BINARY_DIR}/qvac-coreml-link")
    set(probe_srcs "")
    set(objs "")
    set(xlist "")
    foreach (t IN LISTS QLT_TARGETS)
        if (NOT TARGET ${t})
            message(STATUS "test-coreml-link: not registered (${t} is not built)")
            return()
        endif()
        get_target_property(ns ${t} QVAC_COREML_NS)
        get_target_property(rt ${t} QVAC_COREML_RUNTIME)
        if (NOT ns OR NOT rt)
            message(STATUS "test-coreml-link: not registered (${t} has no Core ML runtime; enable its *_COREML option)")
            return()
        endif()
        file(CONFIGURE OUTPUT "${gen}/probe_${ns}.cpp"
             CONTENT "#define QVAC_COREML_NS ${ns}\n#include \"link_probe.inc\"\n" @ONLY)
        list(APPEND probe_srcs "${gen}/probe_${ns}.cpp")
        string(APPEND xlist " X(${ns})")
        list(APPEND objs $<TARGET_OBJECTS:${rt}>)
    endforeach()
    file(CONFIGURE OUTPUT "${gen}/qvac_coreml_link_probes.h"
         CONTENT "#pragma once\n#define QVAC_COREML_LINK_PROBES(X)${xlist}\n" @ONLY)

    # The three engine libraries plus every object of every runtime copy in
    # one image: any runtime symbol that escaped its namespace (or an
    # Objective-C class) is a duplicate and fails the link. Whole-archive
    # loading of the engine libraries is not used: tts-cpp and qvac-parakeet
    # both compile dr_wav's implementation, a pre-existing collision outside
    # the runtime.
    add_executable(test-coreml-link "${dir}/test/test_link.cpp" ${probe_srcs} ${objs})
    target_include_directories(test-coreml-link PRIVATE
        "${gen}" "${dir}/include" "${dir}/src" "${dir}/third_party" "${dir}/test")
    target_compile_features(test-coreml-link PRIVATE cxx_std_17)
    target_link_libraries(test-coreml-link PRIVATE ${QLT_TARGETS})
    add_test(NAME test-coreml-link COMMAND test-coreml-link)
    set_tests_properties(test-coreml-link PROPERTIES LABELS "unit;coreml-runtime" TIMEOUT 120)
endfunction()
