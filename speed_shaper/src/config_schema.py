"""Configuration schema for speed shaper parameters.

This module defines the data structure for saving and loading speed shaper
configurations, enabling transfer of tuned parameters to RL training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class ShaperConfig:
    """Speed shaper configuration for QP-based profile shaping.
    
    This configuration captures all parameters needed to reproduce a speed shaper
    setup, including weights, weight schedules, and constraints.
    
    Attributes:
        dt: Time step size in seconds
        wE_start: Initial error weight
        wE_end: Final error weight
        lamE: Error weight decay/growth rate (1/seconds)
        wA_start: Initial acceleration weight
        wA_end: Final acceleration weight
        lamA: Acceleration weight decay/growth rate (1/seconds)
        wJ_start: Initial jerk weight
        wJ_end: Final jerk weight
        lamJ: Jerk weight decay/growth rate (1/seconds)
        a_min: Minimum acceleration bound (m/s²), None if disabled
        a_max: Maximum acceleration bound (m/s²), None if disabled
        j_min: Minimum jerk bound (m/s³), None if disabled
        j_max: Maximum jerk bound (m/s³), None if disabled
        enable_accel_bounds: Whether acceleration bounds are active
        enable_jerk_bounds: Whether jerk bounds are active
        enable_terminal_constraint: Whether terminal velocity constraint is active
        metadata: Additional metadata (creation time, description, etc.)
    """
    
    dt: float
    wE_start: float
    wE_end: float
    lamE: float
    wA_start: float
    wA_end: float
    lamA: float
    wJ_start: float
    wJ_end: float
    lamJ: float
    a_min: float | None = None
    a_max: float | None = None
    j_min: float | None = None
    j_max: float | None = None
    enable_accel_bounds: bool = False
    enable_jerk_bounds: bool = False
    enable_terminal_constraint: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def to_json(self, path: Path, indent: int = 2) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: Output file path
            indent: JSON indentation level (default: 2)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to metadata if not present
        data = self.to_dict()
        if 'created_at' not in data['metadata']:
            data['metadata']['created_at'] = datetime.now().isoformat()
        
        with path.open('w') as f:
            json.dump(data, f, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShaperConfig:
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration parameters
            
        Returns:
            ShaperConfig instance
            
        Raises:
            ValueError: If required fields are missing
        """
        # Required fields
        required = ['dt', 'wE_start', 'wE_end', 'lamE', 
                   'wA_start', 'wA_end', 'lamA',
                   'wJ_start', 'wJ_end', 'lamJ']
        
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Extract fields, using defaults for optional ones
        return cls(
            dt=float(data['dt']),
            wE_start=float(data['wE_start']),
            wE_end=float(data['wE_end']),
            lamE=float(data['lamE']),
            wA_start=float(data['wA_start']),
            wA_end=float(data['wA_end']),
            lamA=float(data['lamA']),
            wJ_start=float(data['wJ_start']),
            wJ_end=float(data['wJ_end']),
            lamJ=float(data['lamJ']),
            a_min=float(data['a_min']) if data.get('a_min') is not None else None,
            a_max=float(data['a_max']) if data.get('a_max') is not None else None,
            j_min=float(data['j_min']) if data.get('j_min') is not None else None,
            j_max=float(data['j_max']) if data.get('j_max') is not None else None,
            enable_accel_bounds=bool(data.get('enable_accel_bounds', False)),
            enable_jerk_bounds=bool(data.get('enable_jerk_bounds', False)),
            enable_terminal_constraint=bool(data.get('enable_terminal_constraint', False)),
            metadata=dict(data.get('metadata', {})),
        )
    
    @classmethod
    def from_json(cls, path: Path) -> ShaperConfig:
        """Load configuration from JSON file.
        
        Args:
            path: Input file path
            
        Returns:
            ShaperConfig instance
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If JSON is invalid or missing required fields
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with path.open('r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation for debugging."""
        lines = [
            "ShaperConfig:",
            f"  dt: {self.dt}s",
            f"  Error weights: {self.wE_start} → {self.wE_end} (λ={self.lamE})",
            f"  Accel weights: {self.wA_start} → {self.wA_end} (λ={self.lamA})",
            f"  Jerk weights: {self.wJ_start} → {self.wJ_end} (λ={self.lamJ})",
        ]
        
        if self.enable_accel_bounds:
            lines.append(f"  Accel bounds: [{self.a_min}, {self.a_max}] m/s²")
        
        if self.enable_jerk_bounds:
            lines.append(f"  Jerk bounds: [{self.j_min}, {self.j_max}] m/s³")
        
        if self.enable_terminal_constraint:
            lines.append("  Terminal constraint: enabled")
        
        return "\n".join(lines)


__all__ = ['ShaperConfig']
