cmake_minimum_required(VERSION 3.18)
project(sparse_sgemv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(sparse_sgemv
  test/main.cpp
  src/tester.cpp
  src/kernel.cu
)

target_link_libraries(sparse_sgemv PRIVATE 
  CUDA::cudart
  CUDA::cublas
)

set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${PROJECT_SOURCE_DIR}/src/include)

target_include_directories(sparse_sgemv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(sparse_sgemv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)