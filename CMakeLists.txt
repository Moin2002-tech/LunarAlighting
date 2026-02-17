cmake_minimum_required(VERSION 3.10)
project(LunarAlightingRL)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)


find_package(ZeroMQ CONFIG REQUIRED)

find_package(msgpack-c CONFIG REQUIRED)

# Find SDL3 for simulation
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# We'll use libtorch's built-in fmt and manually link spdlog

# Configure doctest


add_executable(LunarAlightingRL main.cpp
        include/Generator/Generator.hpp
        include/Generator/FeedForwardGenerator.hpp
        include/Generator/RecurrentGenerator.hpp
        include/Distribution/Distribution.hpp
        include/Distribution/Normal.hpp
        include/Distribution/Categorical.hpp
        include/Distribution/Burnoulli.hpp
        include/Space.hpp
        include/RunningMeanStd.hpp
        include/ObservationNormalizer.hpp
        include/Storage.hpp
        include/Model/nnBase.hpp
        include/Model/OutputLayers.hpp
        include/Model/policy.hpp
        include/Model/modelUtils.hpp
        include/Model/mlp_base.hpp
        include/Model/CNNBase.hpp
        include/Algorithms/Algorithm.hpp
        include/Algorithms/A2c.hpp
        include/Algorithms/PPO.hpp
        include/LunarAlighting.hpp
        src/third_party/doctest.hpp
        src/third_party/doctest.cpp
        src/Generator/FeedForwardGenerator.cpp
        src/Generator/RecurrentGenerator.cpp
        src/Distribution/Bernoulli.cpp
        src/Distribution/Distribution.cpp
        src/Distribution/Categorical.cpp
        src/Distribution/Normal.cpp
        src/Storage.cpp
        src/RunningMeanStd.cpp
        src/ObservationNormalizer.cpp
        src/Model/mlp_base.cpp
        src/Model/modelUtils.cpp
        src/Model/nnBase.cpp
        src/Model/CNNBase.cpp
        src/Model/OutputLayer.cpp
        src/Model/Policy.cpp
        src/Algorithm/A2c.cpp
        src/Algorithm/PPO.cpp
        LunarAlighting/Communication.hpp
        LunarAlighting/Request.hpp
        LunarAlighting/GymClient.cpp
        LunarAlighting/Communication.cpp
        LunarAlighting/LunarSimulation.cpp
        UnitsTest/UnitsTest.cpp
)


find_package(SDL3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)




find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

target_include_directories(LunarAlightingRL PRIVATE
        "${CMAKE_SOURCE_DIR}/External/libtorch/include"
        "${CMAKE_SOURCE_DIR}/External/libtorch/include/torch/csrc/api/include"

)

target_link_libraries(LunarAlightingRL PRIVATE

        -Wl,--no-as-needed
        -Wl,--whole-archive
        "${CMAKE_SOURCE_DIR}/External/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_SOURCE_DIR}/External/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_SOURCE_DIR}/External/libtorch/lib/libtorch.so"
        -Wl,--no-whole-archive

        "${CMAKE_SOURCE_DIR}/External/libtorch/lib/libc10_cuda.so"
        "${CMAKE_SOURCE_DIR}/External/libtorch/lib/libc10.so"

        # 1. CUDA Runtime (from your toolkit)
        "/usr/local/cuda/lib64/libcudart.so"

        # 2. CUDA Driver (The one usually missing - found in system libs)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

        # 3. NVRTC (Compiler library)
        "/usr/local/cuda/lib64/libnvrtc.so"

        # System base libs
        pthread
        dl
        m
)

# Link fmt and spdlog together to ensure proper dependency order
# Use libtorch's fmt library to avoid version conflicts
target_link_libraries(LunarAlightingRL PRIVATE 
    "${CMAKE_SOURCE_DIR}/External/libtorch/lib/libfmt.a"
    "/home/moinshaikh/vcpkg/installed/x64-linux/debug/lib/libspdlogd.a"
)

target_link_libraries(LunarAlightingRL PRIVATE libzmq-static)

target_link_libraries(LunarAlightingRL PRIVATE msgpack-c)

# Ensure fmt library is properly linked in debug mode
set_target_properties(LunarAlightingRL PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/External/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        MAP_IMPORTED_CONFIG_DEBUG "DEBUG;RELEASE"
)