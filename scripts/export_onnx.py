#!/usr/bin/env python3
"""Export trained SAC policy to ONNX format for C++ inference.

This script loads a trained SAC checkpoint and exports the policy network
to ONNX format, suitable for deployment in C++ applications using ONNX Runtime.

Usage:
    python scripts/export_onnx.py --checkpoint training/checkpoints/latest.pt --output policy.onnx

The script also generates a metadata.json file with model dimensions and
configuration needed for correct inference in C++.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.policy_loader import load_policy_from_checkpoint


class CombinedSysIDPolicy(nn.Module):
    """Combined model that includes both SysID encoder and policy.
    
    For ONNX export, this model:
    1. Takes base observation, speed, action history, and encoder hidden state
    2. Computes z_t using the encoder
    3. Augments observation with z_t
    4. Runs policy to get action
    5. Returns action and new hidden state
    
    Note: Due to FeatureBuilder's implementation, features at step t use the action
    from step t-2 (not t-1). The ONNX model expects the caller to pass actions with
    the correct offset:
    - prev_action: action from step t-2 (used in features)
    - prev_prev_action: action from step t-3 (used for du computation)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_norm: nn.Module,
        policy: nn.Module,
        dt: float,
        base_obs_dim: int,
        z_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_norm = encoder_norm
        self.policy_net = policy.net
        self.mu_head = policy.mu_head
        self.register_buffer("action_scale", policy.action_scale.clone())
        self.register_buffer("action_bias", policy.action_bias.clone())
        self.dt = dt
        self.base_obs_dim = base_obs_dim
        self.z_dim = z_dim
    
    def forward(
        self,
        base_obs: torch.Tensor,
        speed: torch.Tensor,
        prev_action: torch.Tensor,
        prev_speed: torch.Tensor,
        prev_prev_action: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            base_obs: Base observation of shape (batch, base_obs_dim)
            speed: Current speed v_t of shape (batch, 1) or (batch,)
            prev_action: Action from step t-2 (u_{t-2}) of shape (batch, 1) or (batch,)
            prev_speed: Speed from step t-1 (v_{t-1}) of shape (batch, 1) or (batch,)
            prev_prev_action: Action from step t-3 (u_{t-3}) of shape (batch, 1) or (batch,)
            hidden_state: Encoder hidden state of shape (batch, hidden_dim)
        
        Returns:
            Tuple of (action, new_hidden_state):
                - action: Action of shape (batch, action_dim)
                - new_hidden_state: New hidden state of shape (batch, hidden_dim)
        """
        # Ensure correct shapes
        if speed.dim() == 1:
            speed = speed.unsqueeze(-1)
        if prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(-1)
        if prev_speed.dim() == 1:
            prev_speed = prev_speed.unsqueeze(-1)
        if prev_prev_action.dim() == 1:
            prev_prev_action = prev_prev_action.unsqueeze(-1)
        
        # Build features: [v_t, u_{t-2}, dv_t, du_{t-2}]
        # where du_{t-2} = u_{t-2} - u_{t-3}
        dv = (speed - prev_speed) / self.dt
        du_prev = prev_action - prev_prev_action
        
        # Stack features
        features = torch.cat([speed, prev_action, dv, du_prev], dim=-1)  # (batch, 4)
        
        # Normalize features
        features_norm = self.encoder_norm(features, update_stats=False)
        
        # Encoder step
        h_new, z_t = self.encoder.step(features_norm, hidden_state)
        
        # Augment observation with z_t
        obs_aug = torch.cat([base_obs, z_t], dim=-1)  # (batch, base_obs_dim + z_dim)
        
        # Policy forward
        features_policy = self.policy_net(obs_aug)
        mu = self.mu_head(features_policy)
        action = torch.tanh(mu) * self.action_scale + self.action_bias
        
        return action, h_new


class DeterministicPolicyWrapper(nn.Module):
    """Wrapper that exports only the deterministic forward pass.
    
    For deployment, we only need the mean action (deterministic policy),
    not the stochastic sampling used during training. This wrapper
    computes: action = tanh(mu) * action_scale + action_bias
    """
    
    def __init__(self, policy: nn.Module):
        super().__init__()
        # Copy the network components
        self.net = policy.net
        self.mu_head = policy.mu_head
        # Register buffers for action scaling (baked into the model)
        self.register_buffer("action_scale", policy.action_scale.clone())
        self.register_buffer("action_bias", policy.action_bias.clone())
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            
        Returns:
            Action tensor of shape (batch, action_dim)
        """
        features = self.net(obs)
        mu = self.mu_head(features)
        # Deterministic action: tanh squashing + scaling
        action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SAC policy to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for ONNX model. Defaults to checkpoint_name.onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate exported model with ONNX Runtime",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip ONNX Runtime validation",
    )
    return parser.parse_args()


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 13,
    validate: bool = True,
) -> dict:
    """Export a trained policy checkpoint to ONNX format.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        output_path: Path for the output ONNX file
        opset_version: ONNX opset version to use
        validate: Whether to validate with ONNX Runtime
        
    Returns:
        Metadata dictionary with model configuration
    """
    print(f"[export] Loading checkpoint: {checkpoint_path}")
    
    # Load raw checkpoint to get metadata and SysID components
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    sysid_enabled = meta.get("sysid_enabled", False)
    
    # Load the policy and configuration
    policy, env_cfg, horizon = load_policy_from_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    
    # Get obs_dim from policy network (this is the augmented dimension if SysID is enabled)
    obs_dim = policy.net[0].in_features
    action_dim = int(meta.get("policy_action_dim", policy.action_dim))
    env_action_dim = int(meta.get("env_action_dim", 1))
    
    print(f"[export] Model dimensions:")
    print(f"         obs_dim: {obs_dim}")
    print(f"         action_dim: {action_dim}")
    print(f"         env_action_dim: {env_action_dim}")
    print(f"         horizon: {horizon}")
    print(f"         sysid_enabled: {sysid_enabled}")
    
    # Load SysID components if enabled
    if sysid_enabled:
        from src.sysid import ContextEncoder, RunningNorm
        
        config = checkpoint.get("config", {})
        sysid_config = config.get("sysid", {})
        z_dim = int(sysid_config.get("dz", 12))
        gru_hidden = int(sysid_config.get("gru_hidden", 64))
        
        # Create encoder and normalization
        encoder = ContextEncoder(input_dim=4, hidden_dim=gru_hidden, z_dim=z_dim)
        encoder_norm = RunningNorm(dim=4, eps=1e-6, clip=10.0)
        
        # Load weights
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_norm.load_state_dict(checkpoint["encoder_norm"])
        encoder.eval()
        encoder_norm.eval()
        
        # Base observation dim (without z_t) - use meta["obs_dim"] which is the base dimension
        base_obs_dim = int(meta.get("obs_dim", obs_dim - z_dim))
        
        print(f"[export] SysID configuration:")
        print(f"         z_dim: {z_dim}")
        print(f"         gru_hidden: {gru_hidden}")
        print(f"         base_obs_dim: {base_obs_dim}")
        
        # Create combined model
        model = CombinedSysIDPolicy(
            encoder=encoder,
            encoder_norm=encoder_norm,
            policy=policy,
            dt=env_cfg.dt,
            base_obs_dim=base_obs_dim,
            z_dim=z_dim,
        )
        model.eval()
        
        # Create dummy inputs for combined model
        dummy_base_obs = torch.randn(1, base_obs_dim, dtype=torch.float32)
        dummy_speed = torch.randn(1, 1, dtype=torch.float32)
        dummy_prev_action = torch.randn(1, 1, dtype=torch.float32)
        dummy_prev_speed = torch.randn(1, 1, dtype=torch.float32)
        dummy_prev_prev_action = torch.randn(1, 1, dtype=torch.float32)
        dummy_hidden = torch.zeros(1, gru_hidden, dtype=torch.float32)
        
        dummy_inputs = (
            dummy_base_obs,
            dummy_speed,
            dummy_prev_action,
            dummy_prev_speed,
            dummy_prev_prev_action,
            dummy_hidden,
        )
        
        input_names = [
            "base_observation",
            "speed",
            "prev_action",
            "prev_speed",
            "prev_prev_action",
            "hidden_state",
        ]
        output_names = ["action", "new_hidden_state"]
        
        dynamic_axes = {
            "base_observation": {0: "batch_size"},
            "speed": {0: "batch_size"},
            "prev_action": {0: "batch_size"},
            "prev_speed": {0: "batch_size"},
            "prev_prev_action": {0: "batch_size"},
            "hidden_state": {0: "batch_size"},
            "action": {0: "batch_size"},
            "new_hidden_state": {0: "batch_size"},
        }
    else:
        # No SysID: use simple policy wrapper
        model = DeterministicPolicyWrapper(policy)
        model.eval()
        
        dummy_inputs = torch.randn(1, obs_dim, dtype=torch.float32)
        input_names = ["observation"]
        output_names = ["action"]
        dynamic_axes = {
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        }
    
    # Export to ONNX
    print(f"[export] Exporting to ONNX (opset {opset_version})...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"[export] Saved ONNX model to: {output_path}")
    
    # Build metadata
    metadata = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_action_dim": env_action_dim,
        "horizon": horizon,
        "action_low": float(env_cfg.action_low),
        "action_high": float(env_cfg.action_high),
        "dt": float(env_cfg.dt),
        "preview_horizon_s": float(env_cfg.preview_horizon_s),
        "action_scale": policy.action_scale.cpu().numpy().tolist(),
        "action_bias": policy.action_bias.cpu().numpy().tolist(),
        "checkpoint_path": str(checkpoint_path),
        "opset_version": opset_version,
        "sysid_enabled": sysid_enabled,
    }
    
    if sysid_enabled:
        metadata.update({
            "z_dim": z_dim,
            "gru_hidden": gru_hidden,
            "base_obs_dim": base_obs_dim,
        })
    
    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[export] Saved metadata to: {metadata_path}")
    
    # Validate with ONNX Runtime
    if validate:
        print("[export] Validating with ONNX Runtime...")
        try:
            import onnx
            import onnxruntime as ort
            
            # Check ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("[export] ONNX model check passed")
            
            # Run inference comparison
            sess = ort.InferenceSession(str(output_path))
            
            if sysid_enabled:
                # Generate random test inputs for SysID model
                test_base_obs = np.random.randn(1, base_obs_dim).astype(np.float32)
                test_speed = np.random.randn(1, 1).astype(np.float32)
                test_prev_action = np.random.randn(1, 1).astype(np.float32)
                test_prev_speed = np.random.randn(1, 1).astype(np.float32)
                test_prev_prev_action = np.random.randn(1, 1).astype(np.float32)
                test_hidden = np.zeros((1, gru_hidden), dtype=np.float32)
                
                test_inputs = {
                    "base_observation": test_base_obs,
                    "speed": test_speed,
                    "prev_action": test_prev_action,
                    "prev_speed": test_prev_speed,
                    "prev_prev_action": test_prev_prev_action,
                    "hidden_state": test_hidden,
                }
                
                # PyTorch inference
                with torch.no_grad():
                    pt_outputs = model(
                        torch.from_numpy(test_base_obs),
                        torch.from_numpy(test_speed),
                        torch.from_numpy(test_prev_action),
                        torch.from_numpy(test_prev_speed),
                        torch.from_numpy(test_prev_prev_action),
                        torch.from_numpy(test_hidden),
                    )
                    pt_action = pt_outputs[0].numpy()
                    pt_hidden = pt_outputs[1].numpy()
                
                # ONNX Runtime inference
                ort_outputs = sess.run(None, test_inputs)
                ort_action = ort_outputs[0]
                ort_hidden = ort_outputs[1]
                
                # Compare outputs
                max_diff_action = np.abs(pt_action - ort_action).max()
                max_diff_hidden = np.abs(pt_hidden - ort_hidden).max()
                print(f"[export] Max difference (action): {max_diff_action:.2e}")
                print(f"[export] Max difference (hidden_state): {max_diff_hidden:.2e}")
                
                if max_diff_action < 1e-5 and max_diff_hidden < 1e-5:
                    print("[export] Validation PASSED")
                else:
                    print("[export] WARNING: Outputs differ more than expected")
            else:
                # Generate random test inputs for simple policy
                test_obs = np.random.randn(1, obs_dim).astype(np.float32)
                
                # PyTorch inference
                with torch.no_grad():
                    pt_output = model(torch.from_numpy(test_obs)).numpy()
                
                # ONNX Runtime inference
                ort_output = sess.run(None, {"observation": test_obs})[0]
                
                # Compare outputs
                max_diff = np.abs(pt_output - ort_output).max()
                print(f"[export] Max difference between PyTorch and ONNX: {max_diff:.2e}")
                
                if max_diff < 1e-5:
                    print("[export] Validation PASSED")
                else:
                    print("[export] WARNING: Outputs differ more than expected")
                
        except ImportError as e:
            print(f"[export] Skipping validation: {e}")
            print("[export] Install with: pip install onnx onnxruntime")
    
    return metadata


def main() -> None:
    args = parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Default output path
    if args.output is None:
        args.output = args.checkpoint.with_suffix(".onnx")
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        validate=args.validate,
    )
    
    print("[export] Done!")


if __name__ == "__main__":
    main()

