cmake_minimum_required(VERSION 3.20)

# PROJECT_VERSION is synchronized by scripts/sync_version.py (single source of truth).
project(glm52_local
    VERSION 2026.9.2
    DESCRIPTION "GLM-5.2 local full inference engine (BF16, original implementation inspired by colibri)"
    LANGUAGES CXX
)

# ---------- Global settings ----------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CTest)
if(BUILD_TESTING)
    enable_testing()
endif()

# ---------- Options ----------
option(ENABLE_CUDA "Enable CUDA GPU backend (NVIDIA; auto-detected at runtime)" OFF)

# AVX2 is an x86-64-only instruction set; Apple Silicon and ARM Linux build
# cleanly with the scalar/NEON code paths instead.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|em64t)$")
    set(GLM_DEFAULT_AVX2 ON)
else()
    set(GLM_DEFAULT_AVX2 OFF)
endif()
option(ENABLE_AVX2 "Enable AVX2/FMA instruction set (x86-64 only)" ${GLM_DEFAULT_AVX2})

# CUDA backend
if(ENABLE_CUDA)
    enable_language(CUDA)
    # Arch list covers the common local-inference cards (Ampere+); override with
    # -DCMAKE_CUDA_ARCHITECTURES="native" on the machine you target.
    set(CMAKE_CUDA_ARCHITECTURES "86;89;120" CACHE STRING
        "NVIDIA compute-capability list (e.g. 86=RTX 30x0, 89=RTX 40x0, 120=RTX 50x0, or native)")
    if(NOT DEFINED CUDAToolkit_ROOT)
        file(GLOB _cuda_dirs "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.*")
        if(_cuda_dirs)
            list(GET _cuda_dirs 0 CUDAToolkit_ROOT)
        endif()
    endif()
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} @ ${CUDAToolkit_NVCC_EXECUTABLE} (arch=${CMAKE_CUDA_ARCHITECTURES})")
endif()

# ---------- OpenMP ----------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

# ---------- Compile options ----------
# MSVC flags are scoped to CXX sources: CUDA sources (nvcc) must never receive
# raw cl-style flags, otherwise nvcc treats them as additional input files.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    if(ENABLE_AVX2)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    # OpenMP: MSVC built-in /openmp (manually added if find_package fails)
    if(NOT OpenMP_CXX_FOUND)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/openmp>)
        message(STATUS "OpenMP: using MSVC /openmp flag")
    endif()
    # Release optimization
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>)
endif()

# ---------- Source files ----------
file(GLOB_RECURSE GLM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Platform-specific translation units (design doc §11). Each platform gets its
# own mmap backend, async-reader implementation, and Python-tokenizer bridge.
if(WIN32)
    set(GLM_PLATFORM_OTHER
        src/storage/mmap_posix.cpp
        src/storage/iocp_reader_posix.cpp
        src/tokenizer/python_tokenizer_posix.cpp
    )
else()
    set(GLM_PLATFORM_OTHER
        src/storage/mmap_win.cpp
        src/storage/iocp_reader.cpp
        src/tokenizer/python_tokenizer.cpp
    )
endif()
foreach(_src IN LISTS GLM_PLATFORM_OTHER)
    list(REMOVE_ITEM GLM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
endforeach()

file(GLOB_RECURSE GLM_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

set(GLM_ALL_SOURCES ${GLM_SOURCES})
if(ENABLE_CUDA AND GLM_CUDA_SOURCES)
    list(APPEND GLM_ALL_SOURCES ${GLM_CUDA_SOURCES})
    set_source_files_properties(${GLM_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(glm ${GLM_ALL_SOURCES})

target_include_directories(glm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# GLM_* version macros come from src/version.h (managed by scripts/sync_version.py).

if(ENABLE_CUDA)
    target_compile_definitions(glm PRIVATE GLM_ENABLE_CUDA=1)
    target_link_libraries(glm PRIVATE CUDA::cudart)
    if(GLM_CUDA_SOURCES)
        target_compile_definitions(glm PRIVATE GLM_HAVE_CU=1)
    endif()
else()
    target_compile_definitions(glm PRIVATE GLM_ENABLE_CUDA=0)
endif()

# OpenMP linking
if(OpenMP_CXX_FOUND)
    target_link_libraries(glm PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------- Functional self-tests ----------
# glm_tests runs the same self_test module compiled into `glm`; verified in CI.
if(BUILD_TESTING)
    add_executable(glm_tests
        src/self_test.cpp
        src/engine/kv_cache.cpp
        src/engine/placement.cpp
        src/engine/scheduler.cpp
        src/model/config.cpp
        src/model/weight_index.cpp
        src/model/safetensors.cpp
        src/model/glm_forward.cpp
        src/compute/cpu_kernels.cpp
        src/compute/cuda_backend.cpp
        src/runtime/runtime_config.cpp
        src/tokenizer/native_tokenizer.cpp
        tests/run_tests.cpp
    )
    target_include_directories(glm_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    if(WIN32)
        target_sources(glm_tests PRIVATE
            src/storage/mmap_win.cpp
            src/storage/iocp_reader.cpp
            src/tokenizer/python_tokenizer.cpp
        )
    else()
        target_sources(glm_tests PRIVATE
            src/storage/mmap_posix.cpp
            src/storage/iocp_reader_posix.cpp
            src/tokenizer/python_tokenizer_posix.cpp
        )
    endif()
    if(NOT ENABLE_CUDA)
        target_compile_definitions(glm_tests PRIVATE GLM_ENABLE_CUDA=0)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(glm_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME self_test COMMAND glm_tests
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(self_test PROPERTIES ENVIRONMENT
        "GLM_TEST_FIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
endif()

# ---------- Install (optional) ----------
install(TARGETS glm RUNTIME DESTINATION bin)

# ---------- Packaging (CPack) ----------
set(CPACK_PACKAGE_NAME "HummingFlight")
set(CPACK_PACKAGE_VERSION 2026.9.2) # synchronized by scripts/sync_version.py
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GLM-5.2 local full inference engine (BF16)")
set(CPACK_PACKAGE_VENDOR "HummingFlight")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_EXECUTABLES "glm" "HummingFlight")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "HummingFlight-${CPACK_PACKAGE_VERSION}")

# Cross-platform archive generation (works on Windows, macOS, and Linux).
# Output dir: <repo>/dist  (used by scripts/package.py and the release workflow).
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist")
set(CPACK_GENERATOR "ZIP")
if(UNIX AND NOT APPLE)
    # Linux: .zip (plus optional .deb when the tooling is available)
    option(CPACK_DEBIAN "Build .deb package" OFF)
    if(CPACK_DEBIAN AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(CPACK_GENERATOR "ZIP;DEB")
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER "HummingFlight <dev@hummingflight.local>")
    endif()
elseif(APPLE)
    # macOS: DragNDrop .dmg (hdiutil ships with macOS)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "HummingFlight-${CPACK_PACKAGE_VERSION}")
endif()

include(CPack)