/** @file onnx_policy.cpp
 * @brief Implementation of ONNX Runtime inference wrapper for SAC policy
 */

#include "onnx_policy.h"

#include <stdexcept>

SACPolicyInference::SACPolicyInference(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SACPolicy"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Input info
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    
    auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    obs_dim_ = static_cast<size_t>(input_shape[1]);
    
    // Output info
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = output_name.get();
    
    auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    action_dim_ = static_cast<size_t>(output_shape[1]);
}

std::vector<float> SACPolicyInference::infer(const std::vector<float>& observation) {
    if (observation.size() != obs_dim_) {
        throw std::runtime_error(
            "Observation size mismatch: expected " + std::to_string(obs_dim_) +
            ", got " + std::to_string(observation.size())
        );
    }
    
    // Create input tensor
    std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(obs_dim_)};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(observation.data()),
        observation.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    // Run inference
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // Extract output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + action_dim_);
}

std::vector<float> SACPolicyInference::inferBatch(const std::vector<float>& observations, size_t batch_size) {
    if (observations.size() != batch_size * obs_dim_) {
        throw std::runtime_error("Observations size mismatch for batch");
    }
    
    std::array<int64_t, 2> input_shape = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(obs_dim_)
    };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(observations.data()),
        observations.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + batch_size * action_dim_);
}

