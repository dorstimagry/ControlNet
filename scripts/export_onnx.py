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

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.policy_loader import load_policy_from_checkpoint


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
        default=17,
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
    opset_version: int = 17,
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
    
    # Load the policy and configuration
    policy, env_cfg, horizon = load_policy_from_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    
    # Load raw checkpoint to get metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    obs_dim = int(meta.get("obs_dim", policy.net[0].in_features))
    action_dim = int(meta.get("policy_action_dim", policy.action_dim))
    env_action_dim = int(meta.get("env_action_dim", 1))
    
    print(f"[export] Model dimensions:")
    print(f"         obs_dim: {obs_dim}")
    print(f"         action_dim: {action_dim}")
    print(f"         env_action_dim: {env_action_dim}")
    print(f"         horizon: {horizon}")
    
    # Create deterministic wrapper
    wrapper = DeterministicPolicyWrapper(policy)
    wrapper.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)
    
    # Export to ONNX
    print(f"[export] Exporting to ONNX (opset {opset_version})...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
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
    }
    
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
            
            # Generate random test inputs
            test_obs = np.random.randn(1, obs_dim).astype(np.float32)
            
            # PyTorch inference
            with torch.no_grad():
                pt_output = wrapper(torch.from_numpy(test_obs)).numpy()
            
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

