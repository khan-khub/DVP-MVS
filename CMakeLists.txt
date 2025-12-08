cmake_minimum_required(VERSION 3.10)
project(APD LANGUAGES C CXX CUDA)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CUDA REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/local/cuda/include
    .
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------
# IMPORTANT FOR RTX 3050
# --------------------------
set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS}
     -O3
     --use_fast_math
     --maxrregcount=128
     --ptxas-options=-v
     -std=c++11
     -gencode arch=compute_86,code=sm_86")

add_executable(APD
    main.cpp
    APD.cpp
    APD.cu
)

target_link_libraries(APD
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
)
