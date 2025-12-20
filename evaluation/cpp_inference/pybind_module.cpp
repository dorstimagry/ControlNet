/** @file pybind_module.cpp
 * @brief PyBind11 bindings for ONNX Runtime inference
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "onnx_policy.h"

namespace py = pybind11;

PYBIND11_MODULE(_onnx_inference, m) {
    m.doc() = "ONNX Runtime inference for SAC policy (PyBind11 module)";

    py::class_<SACPolicyInference>(m, "SACPolicyInference")
        .def(py::init<const std::string&>(), 
             "Initialize ONNX inference session",
             py::arg("model_path"))
        .def("infer", 
             &SACPolicyInference::infer,
             "Run inference on a single observation",
             py::arg("observation"))
        .def("infer_batch",
             &SACPolicyInference::inferBatch,
             "Run inference on a batch of observations",
             py::arg("observations"),
             py::arg("batch_size"))
        .def("get_obs_dim",
             &SACPolicyInference::getObsDim,
             "Get observation dimension")
        .def("get_action_dim",
             &SACPolicyInference::getActionDim,
             "Get action dimension");
}

