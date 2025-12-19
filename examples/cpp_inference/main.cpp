/**
 * @file main.cpp
 * @brief SAC Policy Inference using ONNX Runtime
 *
 * This example demonstrates how to load and run inference on an exported
 * SAC policy model using ONNX Runtime in C++.
 *
 * Usage:
 *   ./sac_inference <model.onnx> [obs_dim]
 *
 * Example:
 *   ./sac_inference ../../training/checkpoints/latest.onnx 34
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
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
    explicit SACPolicyInference(const std::string& model_path)
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
        
        std::cout << "[SACPolicy] Loaded model: " << model_path << std::endl;
        std::cout << "[SACPolicy] obs_dim: " << obs_dim_ << ", action_dim: " << action_dim_ << std::endl;
    }
    
    /**
     * @brief Run inference on a single observation
     * @param observation Input observation vector (must have obs_dim elements)
     * @return Action vector
     */
    std::vector<float> infer(const std::vector<float>& observation) {
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
    
    /**
     * @brief Run inference on a batch of observations
     * @param observations Flattened batch of observations (batch_size * obs_dim elements)
     * @param batch_size Number of observations in the batch
     * @return Flattened action vectors (batch_size * action_dim elements)
     */
    std::vector<float> inferBatch(const std::vector<float>& observations, size_t batch_size) {
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
    
    size_t getObsDim() const { return obs_dim_; }
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


/**
 * @brief Build a sample observation vector
 * 
 * The observation format is:
 *   [current_speed, prev_speed, prev_prev_speed, prev_action, ref_speeds...]
 * 
 * Where ref_speeds are the target speeds for the preview horizon.
 */
std::vector<float> buildSampleObservation(size_t obs_dim) {
    std::vector<float> obs(obs_dim);
    
    // Example: vehicle at 10 m/s with constant target of 15 m/s
    float current_speed = 10.0f;
    float prev_speed = 9.8f;
    float prev_prev_speed = 9.6f;
    float prev_action = 0.5f;
    
    obs[0] = current_speed;
    obs[1] = prev_speed;
    obs[2] = prev_prev_speed;
    obs[3] = prev_action;
    
    // Fill reference speeds (target profile)
    float target_speed = 15.0f;
    for (size_t i = 4; i < obs_dim; ++i) {
        obs[i] = target_speed;
    }
    
    return obs;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [obs_dim]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  model.onnx  Path to the exported ONNX model" << std::endl;
        std::cerr << "  obs_dim     (optional) Override observation dimension for sample input" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    try {
        // Create inference session
        SACPolicyInference policy(model_path);
        
        // Use model's obs_dim or override from command line
        size_t obs_dim = policy.getObsDim();
        if (argc >= 3) {
            obs_dim = std::stoul(argv[2]);
        }
        
        // Build sample observation
        std::vector<float> observation = buildSampleObservation(obs_dim);
        
        std::cout << "\n[Test] Running single inference..." << std::endl;
        std::cout << "[Test] Input observation (first 8 elements): ";
        for (size_t i = 0; i < std::min(size_t(8), observation.size()); ++i) {
            std::cout << observation[i] << " ";
        }
        std::cout << "..." << std::endl;
        
        // Single inference
        auto action = policy.infer(observation);
        
        std::cout << "[Test] Output action: ";
        for (float a : action) {
            std::cout << a << " ";
        }
        std::cout << std::endl;
        
        // First element is the immediate action to apply
        std::cout << "[Test] Immediate action (throttle/brake): " << action[0] << std::endl;
        
        // Benchmark
        std::cout << "\n[Benchmark] Running 1000 inferences..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        const int num_iterations = 1000;
        for (int i = 0; i < num_iterations; ++i) {
            auto _ = policy.infer(observation);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double avg_time_us = static_cast<double>(duration_us) / num_iterations;
        double throughput = 1e6 / avg_time_us;
        
        std::cout << "[Benchmark] Average inference time: " << avg_time_us << " us" << std::endl;
        std::cout << "[Benchmark] Throughput: " << throughput << " inferences/sec" << std::endl;
        
        // Batch inference test
        std::cout << "\n[Test] Running batch inference (batch_size=16)..." << std::endl;
        
        size_t batch_size = 16;
        std::vector<float> batch_obs;
        batch_obs.reserve(batch_size * obs_dim);
        for (size_t b = 0; b < batch_size; ++b) {
            auto obs = buildSampleObservation(obs_dim);
            obs[0] = 5.0f + static_cast<float>(b);  // Vary speed
            batch_obs.insert(batch_obs.end(), obs.begin(), obs.end());
        }
        
        auto batch_actions = policy.inferBatch(batch_obs, batch_size);
        
        std::cout << "[Test] Batch output (first action of each sample): ";
        for (size_t b = 0; b < batch_size; ++b) {
            std::cout << batch_actions[b * policy.getActionDim()] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\n[Done] Inference test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

