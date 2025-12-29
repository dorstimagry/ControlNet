/** @file onnx_policy.cpp
 * @brief Implementation of ONNX Runtime inference wrapper for SAC policy
 */

#include "onnx_policy.h"

#include <stdexcept>

SACPolicyInference::SACPolicyInference(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SACPolicy"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      is_sysid_model_(false),
      hidden_dim_(0) {
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t num_inputs = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();
    
    // Check if this is a SysID model (has multiple inputs)
    is_sysid_model_ = (num_inputs > 1);
    
    if (is_sysid_model_) {
        // SysID model: multiple inputs and outputs
        input_names_.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.get());
        }
        
        output_names_.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.get());
        }
        
        // Get dimensions from first input (base_observation) and outputs
        auto base_obs_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        obs_dim_ = static_cast<size_t>(base_obs_shape[1]);
        
        auto action_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        action_dim_ = static_cast<size_t>(action_shape[1]);
        
        // Get hidden state dimension from last input
        auto hidden_shape = session_->GetInputTypeInfo(5).GetTensorTypeAndShapeInfo().GetShape();
        hidden_dim_ = static_cast<size_t>(hidden_shape[1]);
        
        // Set legacy names for backward compatibility
        input_name_ = input_names_[0];
        output_name_ = output_names_[0];
    } else {
        // Non-SysID model: single input/output
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();
        
        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        obs_dim_ = static_cast<size_t>(input_shape[1]);
        
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name.get();
        
        auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        action_dim_ = static_cast<size_t>(output_shape[1]);
    }
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

std::pair<std::vector<float>, std::vector<float>> SACPolicyInference::inferSysID(
    const std::vector<float>& base_obs,
    float speed,
    float prev_action,
    float prev_speed,
    float prev_prev_action,
    const std::vector<float>& hidden_state
) {
    if (!is_sysid_model_) {
        throw std::runtime_error("inferSysID called on non-SysID model");
    }
    
    if (base_obs.size() != obs_dim_) {
        throw std::runtime_error(
            "Base observation size mismatch: expected " + std::to_string(obs_dim_) +
            ", got " + std::to_string(base_obs.size())
        );
    }
    
    if (hidden_state.size() != hidden_dim_) {
        throw std::runtime_error(
            "Hidden state size mismatch: expected " + std::to_string(hidden_dim_) +
            ", got " + std::to_string(hidden_state.size())
        );
    }
    
    // Prepare input tensors
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names_cstr;
    
    // base_observation
    std::array<int64_t, 2> base_obs_shape = {1, static_cast<int64_t>(obs_dim_)};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(base_obs.data()),
        base_obs.size(),
        base_obs_shape.data(),
        base_obs_shape.size()
    ));
    input_names_cstr.push_back(input_names_[0].c_str());
    
    // speed
    std::array<int64_t, 2> speed_shape = {1, 1};
    float speed_val = speed;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        &speed_val,
        1,
        speed_shape.data(),
        speed_shape.size()
    ));
    input_names_cstr.push_back(input_names_[1].c_str());
    
    // prev_action
    float prev_action_val = prev_action;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        &prev_action_val,
        1,
        speed_shape.data(),
        speed_shape.size()
    ));
    input_names_cstr.push_back(input_names_[2].c_str());
    
    // prev_speed
    float prev_speed_val = prev_speed;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        &prev_speed_val,
        1,
        speed_shape.data(),
        speed_shape.size()
    ));
    input_names_cstr.push_back(input_names_[3].c_str());
    
    // prev_prev_action
    float prev_prev_action_val = prev_prev_action;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        &prev_prev_action_val,
        1,
        speed_shape.data(),
        speed_shape.size()
    ));
    input_names_cstr.push_back(input_names_[4].c_str());
    
    // hidden_state
    std::array<int64_t, 2> hidden_shape = {1, static_cast<int64_t>(hidden_dim_)};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(hidden_state.data()),
        hidden_state.size(),
        hidden_shape.data(),
        hidden_shape.size()
    ));
    input_names_cstr.push_back(input_names_[5].c_str());
    
    // Prepare output names
    std::vector<const char*> output_names_cstr;
    for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
        output_names_cstr.data(), output_names_cstr.size()
    );
    
    // Extract outputs: action and new_hidden_state
    float* action_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> action(action_data, action_data + action_dim_);
    
    float* hidden_data = output_tensors[1].GetTensorMutableData<float>();
    std::vector<float> new_hidden_state(hidden_data, hidden_data + hidden_dim_);
    
    return std::make_pair(action, new_hidden_state);
}

