/** @file onnx_policy.h
 * @brief ONNX Runtime inference wrapper for SAC policy
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

/**
 * @class SACPolicyInference
 * @brief Wrapper class for ONNX Runtime inference of SAC policy
 */
class SACPolicyInference {
public:
    /**
     * @brief Construct inference session from ONNX model file
     * @param model_path Path to the ONNX model file
     */
    explicit SACPolicyInference(const std::string& model_path);
    
    /**
     * @brief Run inference on a single observation
     * @param observation Input observation vector (must have obs_dim elements)
     * @return Action vector
     */
    std::vector<float> infer(const std::vector<float>& observation);
    
    /**
     * @brief Run inference on a batch of observations
     * @param observations Flattened batch of observations (batch_size * obs_dim elements)
     * @param batch_size Number of observations in the batch
     * @return Flattened action vectors (batch_size * action_dim elements)
     */
    std::vector<float> inferBatch(const std::vector<float>& observations, size_t batch_size);
    
    /**
     * @brief Get observation dimension
     * @return Observation dimension
     */
    size_t getObsDim() const { return obs_dim_; }
    
    /**
     * @brief Get action dimension
     * @return Action dimension
     */
    size_t getActionDim() const { return action_dim_; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::string input_name_;
    std::string output_name_;
    size_t obs_dim_;
    size_t action_dim_;
};

