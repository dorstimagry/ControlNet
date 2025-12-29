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
            
            # Check if SysID methods are available (C++ module might need rebuilding)
            if hasattr(self._policy, 'is_sysid_model'):
                self._is_sysid = self._policy.is_sysid_model()
                self._hidden_dim = self._policy.get_hidden_dim() if self._is_sysid else 0
            else:
                # Fallback: assume non-SysID model if methods not available
                # This happens if C++ module hasn't been rebuilt with SysID support
                self._is_sysid = False
                self._hidden_dim = 0
                import warnings
                warnings.warn(
                    "C++ ONNX inference module doesn't have SysID support. "
                    "Please rebuild the C++ module to enable SysID:\n"
                    "  cd evaluation/cpp_inference\n"
                    "  mkdir -p build && cd build\n"
                    "  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..\n"
                    "  cmake --build .\n"
                    "\nFalling back to non-SysID inference mode.",
                    UserWarning
                )
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
    
    @property
    def is_sysid_model(self) -> bool:
        """Check if this is a SysID model."""
        return self._is_sysid
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden state dimension (for SysID models)."""
        return self._hidden_dim
    
    def infer_sysid(
        self,
        base_obs: np.ndarray,
        speed: float,
        prev_action: float,
        prev_speed: float,
        prev_prev_action: float,
        hidden_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference with SysID model.
        
        Args:
            base_obs: Base observation (without z_t) of shape (base_obs_dim,)
            speed: Current speed (scalar)
            prev_action: Previous action (scalar)
            prev_speed: Previous speed (scalar)
            prev_prev_action: Previous previous action (scalar)
            hidden_state: Encoder hidden state of shape (hidden_dim,)
            
        Returns:
            Tuple of (action, new_hidden_state):
                - action: Action array of shape (action_dim,)
                - new_hidden_state: New hidden state array of shape (hidden_dim,)
        """
        if not self._is_sysid:
            raise ValueError("infer_sysid called on non-SysID model")
        
        base_obs_arr = np.asarray(base_obs, dtype=np.float32)
        if base_obs_arr.ndim != 1 or base_obs_arr.shape[0] != self._obs_dim:
            raise ValueError(
                f"Expected base_obs shape ({self._obs_dim},), got {base_obs_arr.shape}"
            )
        
        hidden_arr = np.asarray(hidden_state, dtype=np.float32)
        if hidden_arr.ndim != 1 or hidden_arr.shape[0] != self._hidden_dim:
            raise ValueError(
                f"Expected hidden_state shape ({self._hidden_dim},), got {hidden_arr.shape}"
            )
        
        # Convert to lists for C++ interface
        base_obs_list = base_obs_arr.tolist()
        hidden_list = hidden_arr.tolist()
        
        # Run inference
        action_list, new_hidden_list = self._policy.infer_sysid(
            base_obs_list,
            float(speed),
            float(prev_action),
            float(prev_speed),
            float(prev_prev_action),
            hidden_list
        )
        
        # Convert back to numpy arrays
        action = np.array(action_list, dtype=np.float32)
        new_hidden = np.array(new_hidden_list, dtype=np.float32)
        
        return action, new_hidden

