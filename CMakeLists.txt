set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "nvcc compiler")

cmake_minimum_required(VERSION 3.18)
project(sparse_sgemv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC_FILES
  src/*.cpp
  src/*.cu
  test/*.cpp
)

add_executable(sparse_sgemv ${SRC_FILES})

target_link_libraries(sparse_sgemv PRIVATE 
  CUDA::cudart
  CUDA::cublas
)

target_compile_options(sparse_sgemv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-g>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${PROJECT_SOURCE_DIR}/src/include)

target_include_directories(sparse_sgemv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(sparse_sgemv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sparse_sgemv PROPERTIES CUDA_ARCHITECTURES "86")