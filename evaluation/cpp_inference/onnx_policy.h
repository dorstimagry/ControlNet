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
     * @brief Run inference on a single observation (non-SysID model)
     * @param observation Input observation vector (must have obs_dim elements)
     * @return Action vector
     */
    std::vector<float> infer(const std::vector<float>& observation);
    
    /**
     * @brief Run inference on a batch of observations (non-SysID model)
     * @param observations Flattened batch of observations (batch_size * obs_dim elements)
     * @param batch_size Number of observations in the batch
     * @return Flattened action vectors (batch_size * action_dim elements)
     */
    std::vector<float> inferBatch(const std::vector<float>& observations, size_t batch_size);
    
    /**
     * @brief Run inference with SysID model (multiple inputs/outputs)
     * @param base_obs Base observation (without z_t)
     * @param speed Current speed
     * @param prev_action Previous action
     * @param prev_speed Previous speed
     * @param prev_prev_action Previous previous action
     * @param hidden_state Encoder hidden state
     * @return Pair of (action, new_hidden_state)
     */
    std::pair<std::vector<float>, std::vector<float>> inferSysID(
        const std::vector<float>& base_obs,
        float speed,
        float prev_action,
        float prev_speed,
        float prev_prev_action,
        const std::vector<float>& hidden_state
    );
    
    /**
     * @brief Get observation dimension
     * @return Observation dimension (for non-SysID) or base observation dimension (for SysID)
     */
    size_t getObsDim() const { return obs_dim_; }
    
    /**
     * @brief Get action dimension
     * @return Action dimension
     */
    size_t getActionDim() const { return action_dim_; }
    
    /**
     * @brief Check if this is a SysID model
     * @return True if model has multiple inputs (SysID), false otherwise
     */
    bool isSysIDModel() const { return is_sysid_model_; }
    
    /**
     * @brief Get hidden state dimension (for SysID models)
     * @return Hidden state dimension, or 0 if not SysID model
     */
    size_t getHiddenDim() const { return hidden_dim_; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::string input_name_;
    std::string output_name_;
    size_t obs_dim_;
    size_t action_dim_;
    
    // SysID model support
    bool is_sysid_model_;
    size_t hidden_dim_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

