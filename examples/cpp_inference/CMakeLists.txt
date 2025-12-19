cmake_minimum_required(VERSION 3.14)
project(sac_policy_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime
# Set ONNXRUNTIME_ROOT to your ONNX Runtime installation path if not in system paths
# e.g., cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
if(DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
else()
    # Try to find system-installed ONNX Runtime
    find_path(ONNXRUNTIME_INCLUDE_DIRS
        NAMES onnxruntime_cxx_api.h
        PATHS
            /usr/include/onnxruntime
            /usr/local/include/onnxruntime
            /opt/onnxruntime/include
    )
    find_library(ONNXRUNTIME_LIBRARIES
        NAMES onnxruntime
        PATHS
            /usr/lib
            /usr/local/lib
            /opt/onnxruntime/lib
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR 
        "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT to your installation path.\n"
        "Example: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime .."
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")

# Create the inference executable
add_executable(sac_inference main.cpp)

target_include_directories(sac_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(sac_inference PRIVATE ${ONNXRUNTIME_LIBRARIES})

# Optional: Install target
install(TARGETS sac_inference DESTINATION bin)

