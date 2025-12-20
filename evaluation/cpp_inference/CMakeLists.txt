cmake_minimum_required(VERSION 3.14)
project(onnx_inference_pybind VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyBind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try to fetch PyBind11 if not found
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find ONNX Runtime
# Set ONNXRUNTIME_ROOT to your ONNX Runtime installation path if not in system paths
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

message(STATUS "PyBind11 found: ${pybind11_VERSION}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")

# Create the PyBind11 module
pybind11_add_module(_onnx_inference
    pybind_module.cpp
    onnx_policy.cpp
)

target_include_directories(_onnx_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(_onnx_inference PRIVATE ${ONNXRUNTIME_LIBRARIES})

# Set output directory to the same directory as the Python module
set_target_properties(_onnx_inference PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
)

