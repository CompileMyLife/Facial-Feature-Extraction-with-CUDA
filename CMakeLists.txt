cmake_minimum_required(VERSION 3.18)
project(FacialFeatureExtractionWithCUDA LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories for headers.
include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# List of source files.
set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu
    ${PROJECT_SOURCE_DIR}/src/main_cuda.cpp
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones/image.c
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones/haar.cpp
    ${PROJECT_SOURCE_DIR}/third_party/viola-jones/rectangles.cpp
)

# Create the executable.
add_executable(${PROJECT_NAME} ${SRC_FILES})

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++14 /Zi /Od>
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G -arch=sm_86>
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_options(${PROJECT_NAME} PRIVATE "/MACHINE:X64")
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14 -g -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS>
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G -arch=sm_86 -Wno-deprecated-gpu-targets>
    )
endif()
