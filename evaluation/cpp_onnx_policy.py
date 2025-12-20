"""Python wrapper for C++ ONNX inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from evaluation.cpp_inference import SACPolicyInference as _CppOnnxPolicy
except ImportError as e:
    _CppOnnxPolicy = None
    _import_error = e


class CppOnnxPolicy:
    """Python wrapper for C++ ONNX Runtime inference.
    
    Provides a numpy array interface compatible with the Python policy interface.
    """
    
    def __init__(self, onnx_model_path: str | Path):
        """Initialize C++ ONNX inference.
        
        Args:
            onnx_model_path: Path to the ONNX model file
            
        Raises:
            ImportError: If the C++ module is not built
            FileNotFoundError: If the ONNX model file doesn't exist
            RuntimeError: If ONNX model loading fails
        """
        if _CppOnnxPolicy is None:
            raise ImportError(
                "C++ ONNX inference module not available. Please build it first:\n"
                "  cd evaluation/cpp_inference\n"
                "  mkdir -p build && cd build\n"
                "  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..\n"
                "  cmake --build .\n"
                f"\nOriginal error: {_import_error}"
            ) from _import_error
        
        onnx_path = Path(onnx_model_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        try:
            self._policy = _CppOnnxPolicy(str(onnx_path))
            self._obs_dim = self._policy.get_obs_dim()
            self._action_dim = self._policy.get_action_dim()
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e
    
    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on a single observation.
        
        Args:
            observation: Observation array of shape (obs_dim,) or (1, obs_dim)
            
        Returns:
            Action array of shape (action_dim,)
            
        Raises:
            ValueError: If observation shape is incorrect
        """
        obs = np.asarray(observation, dtype=np.float32)
        
        # Handle both (obs_dim,) and (1, obs_dim) shapes
        if obs.ndim == 2:
            if obs.shape[0] != 1:
                raise ValueError(
                    f"Expected observation shape (obs_dim,) or (1, obs_dim), "
                    f"got {obs.shape}"
                )
            obs = obs[0]
        elif obs.ndim != 1:
            raise ValueError(
                f"Expected observation shape (obs_dim,) or (1, obs_dim), "
                f"got {obs.shape}"
            )
        
        if obs.shape[0] != self._obs_dim:
            raise ValueError(
                f"Observation dimension mismatch: expected {self._obs_dim}, "
                f"got {obs.shape[0]}"
            )
        
        # Convert to list for C++ interface
        obs_list = obs.tolist()
        
        # Run inference
        action_list = self._policy.infer(obs_list)
        
        # Convert back to numpy array
        return np.array(action_list, dtype=np.float32)
    
    @property
    def obs_dim(self) -> int:
        """Get observation dimension."""
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return self._action_dim

