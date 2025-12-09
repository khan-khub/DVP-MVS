cmake_minimum_required(VERSION 3.10)
project(APD LANGUAGES C CXX CUDA)

# ---- CMake Policies (Fix all warnings) ----
cmake_policy(SET CMP0104 NEW)   # CUDA architectures rule
cmake_policy(SET CMP0146 NEW)   # FindCUDA module removed
cmake_policy(SET CMP0167 NEW)   # FindBoost legacy lookup

# ---- CUDA Architecture: RTX 5080 (Blackwell, SM 90) ----
set(CMAKE_CUDA_ARCHITECTURES 90)

# ---- Boost / OpenCV ----
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# ---- Include Paths ----
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    /usr/local/cuda/include
    .
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CUDA NVCC Flags (optimized for RTX 5080) ----
set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS}
     -O3
     --use_fast_math
     --maxrregcount=128
     --ptxas-options=-v
     -std=c++11
     -gencode arch=compute_90,code=sm_90"
)

# ---- Build APD ----
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
