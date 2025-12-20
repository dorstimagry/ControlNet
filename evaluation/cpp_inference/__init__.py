"""C++ ONNX inference module for SAC policy evaluation."""

from __future__ import annotations

try:
    from evaluation.cpp_inference._onnx_inference import SACPolicyInference
    __all__ = ["SACPolicyInference"]
except ImportError as e:
    # Module not built yet - provide helpful error message
    import sys
    from pathlib import Path
    
    module_path = Path(__file__).parent / "_onnx_inference.so"
    if not module_path.exists():
        raise ImportError(
            f"C++ ONNX inference module not found. Please build it first:\n"
            f"  cd {Path(__file__).parent}\n"
            f"  mkdir -p build && cd build\n"
            f"  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..\n"
            f"  cmake --build .\n"
            f"\nOriginal error: {e}"
        ) from e
    raise

