"""Map grid discretization for (u, v) space."""

from __future__ import annotations

import numpy as np


class MapGrid:
    """Discretize (u, v) space for acceleration map representation.
    
    The grid divides:
    - u (signed command) into N_u bins over [-1, 1]
    - v (speed) into N_v bins over [0, v_max]
    
    Provides binning and center computation utilities.
    """
    
    def __init__(self, N_u: int = 100, N_v: int = 100, v_max: float = 30.0):
        """Initialize map grid.
        
        Args:
            N_u: Number of bins for u (signed command)
            N_v: Number of bins for v (speed)
            v_max: Maximum speed (m/s)
        """
        if N_u < 2:
            raise ValueError(f"N_u must be >= 2, got {N_u}")
        if N_v < 2:
            raise ValueError(f"N_v must be >= 2, got {N_v}")
        if v_max <= 0:
            raise ValueError(f"v_max must be positive, got {v_max}")
        
        self.N_u = N_u
        self.N_v = N_v
        self.v_max = v_max
        
        # Create bin edges
        self.u_edges = np.linspace(-1.0, 1.0, N_u + 1)
        self.v_edges = np.linspace(0.0, v_max, N_v + 1)
        
        # Precompute bin centers
        self.u_centers = (self.u_edges[:-1] + self.u_edges[1:]) / 2.0
        self.v_centers = (self.v_edges[:-1] + self.v_edges[1:]) / 2.0
    
    def bin_u(self, u: float | np.ndarray) -> int | np.ndarray:
        """Map command u to bin index, clamping to [0, N_u-1].
        
        Args:
            u: Command value(s) in [-1, 1]
        
        Returns:
            Bin index or indices
        """
        # Find bin using searchsorted
        indices = np.searchsorted(self.u_edges, u, side='right') - 1
        # Clamp to valid range
        indices = np.clip(indices, 0, self.N_u - 1)
        
        # Return scalar if input was scalar
        if np.isscalar(u):
            return int(indices)
        return indices.astype(np.int32)
    
    def bin_v(self, v: float | np.ndarray) -> int | np.ndarray:
        """Map speed v to bin index, clamping to [0, N_v-1].
        
        Args:
            v: Speed value(s) in [0, v_max]
        
        Returns:
            Bin index or indices
        """
        # Find bin using searchsorted
        indices = np.searchsorted(self.v_edges, v, side='right') - 1
        # Clamp to valid range
        indices = np.clip(indices, 0, self.N_v - 1)
        
        # Return scalar if input was scalar
        if np.isscalar(v):
            return int(indices)
        return indices.astype(np.int32)
    
    def center_u(self, i_u: int | np.ndarray) -> float | np.ndarray:
        """Get bin center for u bin index.
        
        Args:
            i_u: Bin index or indices
        
        Returns:
            Center value(s)
        """
        return self.u_centers[i_u]
    
    def center_v(self, i_v: int | np.ndarray) -> float | np.ndarray:
        """Get bin center for v bin index.
        
        Args:
            i_v: Bin index or indices
        
        Returns:
            Center value(s)
        """
        return self.v_centers[i_v]
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return map shape (N_u, N_v)."""
        return (self.N_u, self.N_v)
    
    @property
    def u_values(self) -> np.ndarray:
        """Return array of u bin centers."""
        return self.u_centers
    
    @property
    def v_values(self) -> np.ndarray:
        """Return array of v bin centers."""
        return self.v_centers

