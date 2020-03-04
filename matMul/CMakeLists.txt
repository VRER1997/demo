cmake_minimum_required(VERSION 3.15)
project(matMul)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
enable_language(CUDA)

add_executable(matMul_cpu matMul.cpp)

cuda_add_executable(matMul_gpu matMul.cu OPTIONS -arch=sm_50)
target_link_libraries(matMul_gpu ${CUDA_LIBRARIES})