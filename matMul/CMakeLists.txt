cmake_minimum_required(VERSION 3.15)
project(matMul)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /opt/cuda)

find_package(CUDA REQUIRED)
enable_language(CUDA)

find_package(Threads REQUIRED)

add_executable(matMulThread matMulThread.cpp)
target_link_libraries(matMulThread ${CMAKE_THREAD_LIBS_INIT})

cuda_add_executable(matMul_gpu matMul.cu OPTIONS -arch=sm_50)
target_link_libraries(matMul_gpu ${CUDA_LIBARARIES} ${CMAKE_THREAD_LIBS_INIT})

