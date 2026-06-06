cmake_minimum_required(VERSION 3.10)
project(mecantensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenMP for acceleration
find_package(OpenMP REQUIRED)
find_package(BLAS)

# Include directories
include_directories(include include/mecan)

# Source Files
set(MECAN_SOURCES
    src/core/allocator.cpp
    src/tensor.cpp
    src/ops/math.cpp
    src/ops/conv.cpp
    src/io/serialization.cpp
    src/autograd/variable.cpp
    src/autograd/engine.cpp
    src/autograd/ops_backward.cpp
    src/python_bridge.cpp
    src/optim/lgc.cpp
    src/ops/bitlinear.cpp
    src/ops/attention.cpp
    src/ops/spatial.cpp
    src/qsbits/qsbits.cpp
    src/midbits/midbits.cpp
    src/nn/linear.cpp
    src/nn/loss.cpp
    src/nn/ntm.cpp
    src/trainer.cpp
    src/vision/quantum_vision.cpp
    src/hal/registry.cpp
    src/runtime/pager.cpp
    src/runtime/backend_registry.cpp
    src/runtime/quant_runtime.cpp
    src/runtime/scheduler.cpp
    src/distributed/collective.cpp
    src/ops/pool.cpp
    src/ops/norm.cpp
    src/ops/upsample.cpp
    src/ops/detect_utils.cpp
    src/ops/pad.cpp
    src/vision/object_engine.cpp
    src/vision/color_engine.cpp
    src/vision/motion_engine.cpp
    src/vision/vector_engine.cpp
    src/vision/light_engine.cpp
)

# Shared Library (For Python ctypes bridge)
add_library(mecantensor SHARED ${MECAN_SOURCES})
target_link_libraries(mecantensor PRIVATE OpenMP::OpenMP_CXX)
if (BLAS_FOUND)
    target_link_libraries(mecantensor PRIVATE ${BLAS_LIBRARIES})
    target_include_directories(mecantensor PRIVATE ${BLAS_INCLUDE_DIRS})
endif()

# Demo Executable
add_executable(mecantensor_demo demo/main.cpp)
target_link_libraries(mecantensor_demo PRIVATE mecantensor OpenMP::OpenMP_CXX)
if (BLAS_FOUND)
    target_link_libraries(mecantensor_demo PRIVATE ${BLAS_LIBRARIES})
endif()

# Optional benchmark targets (some repositories keep these under benchmarks/)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/demo/test_bench_1.cpp)
    add_executable(mecantensor_testbench_1 demo/test_bench_1.cpp)
    target_link_libraries(mecantensor_testbench_1 PRIVATE mecantensor OpenMP::OpenMP_CXX)
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/test_bench_2.cpp)
    add_executable(mecantensor_testbench_2 benchmarks/test_bench_2.cpp)
    target_link_libraries(mecantensor_testbench_2 PRIVATE mecantensor OpenMP::OpenMP_CXX)
    if (BLAS_FOUND)
        target_link_libraries(mecantensor_testbench_2 PRIVATE ${BLAS_LIBRARIES})
    endif()
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/test_runtime_core.cpp)
    add_executable(mecantensor_runtime_core_test benchmarks/test_runtime_core.cpp)
    target_link_libraries(mecantensor_runtime_core_test PRIVATE mecantensor OpenMP::OpenMP_CXX)
endif()
