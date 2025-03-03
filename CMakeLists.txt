cmake_minimum_required(VERSION 3.18)

# Declare the project with C, C++ and CUDA languages.
project(FacialFeatureExtractionWithCUDA LANGUAGES C CXX CUDA)

# Set the C++ and CUDA standards.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find the CUDA Toolkit (ensures CUDA_INCLUDE_DIRS is set).
find_package(CUDA REQUIRED)

# Include directories.
include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# List your source files.
set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu
    ${PROJECT_SOURCE_DIR}/src/main_cuda.cpp
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones/image.c
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones/haar.cpp
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones/rectangles.cpp
)

# Create the executable target.
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Set compile options depending on the compiler.
if(MSVC)
    # For C++ with MSVC:
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++14 /Zi /Od>
    )
    # For CUDA on Windows, add debug flags and specify target architecture for Nvidia 3060 Ti.
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G -arch=sm_86>
    )
    # Suppress warnings about unsafe functions.
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    # For non-MSVC systems (GCC/Clang).
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14 -g -G -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS>
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G --sanitize=address -arch=sm_86 -Wno-deprecated-gpu-targets>
    )
endif()

# Set linker options (force 64-bit machine).
target_link_options(${PROJECT_NAME} PRIVATE "/MACHINE:X64")
