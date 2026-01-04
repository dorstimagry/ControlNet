"""Unit tests for MapGrid."""

import numpy as np
import pytest

from src.maps.grid import MapGrid


def test_grid_initialization():
    """Test basic initialization."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    assert grid.N_u == 100
    assert grid.N_v == 100
    assert grid.v_max == 30.0
    assert grid.shape == (100, 100)


def test_grid_invalid_params():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="N_u must be >= 2"):
        MapGrid(N_u=1, N_v=100)
    
    with pytest.raises(ValueError, match="N_v must be >= 2"):
        MapGrid(N_u=100, N_v=1)
    
    with pytest.raises(ValueError, match="v_max must be positive"):
        MapGrid(N_u=100, N_v=100, v_max=-1.0)


def test_bin_u_boundary_cases():
    """Test u binning at boundaries."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Minimum value should map to bin 0
    assert grid.bin_u(-1.0) == 0
    
    # Maximum value should map to last bin
    assert grid.bin_u(1.0) == 40
    
    # Out of range should clamp
    assert grid.bin_u(-2.0) == 0
    assert grid.bin_u(2.0) == 40


def test_bin_v_boundary_cases():
    """Test v binning at boundaries."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Minimum value should map to bin 0
    assert grid.bin_v(0.0) == 0
    
    # Maximum value should map to last bin
    assert grid.bin_v(30.0) == 99
    
    # Out of range should clamp
    assert grid.bin_v(-1.0) == 0
    assert grid.bin_v(50.0) == 99


def test_bin_u_monotonic():
    """Test that u binning is monotonic."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    u_vals = np.linspace(-1.0, 1.0, 100)
    bins = grid.bin_u(u_vals)
    
    # Bins should be non-decreasing
    assert np.all(np.diff(bins) >= 0)


def test_bin_v_monotonic():
    """Test that v binning is monotonic."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    v_vals = np.linspace(0.0, 30.0, 100)
    bins = grid.bin_v(v_vals)
    
    # Bins should be non-decreasing
    assert np.all(np.diff(bins) >= 0)


def test_center_roundtrip():
    """Test that bin centers map back to their own bins."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # For all u bins
    for i_u in range(grid.N_u):
        u_center = grid.center_u(i_u)
        assert grid.bin_u(u_center) == i_u
    
    # For all v bins
    for i_v in range(grid.N_v):
        v_center = grid.center_v(i_v)
        assert grid.bin_v(v_center) == i_v


def test_centers_inside_edges():
    """Test that bin centers are inside bin edges."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Check u centers
    for i_u in range(grid.N_u):
        u_center = grid.center_u(i_u)
        assert grid.u_edges[i_u] <= u_center <= grid.u_edges[i_u + 1]
    
    # Check v centers
    for i_v in range(grid.N_v):
        v_center = grid.center_v(i_v)
        assert grid.v_edges[i_v] <= v_center <= grid.v_edges[i_v + 1]


def test_vectorized_binning():
    """Test that binning works with arrays."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Test u binning with array
    u_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    bins_u = grid.bin_u(u_vals)
    assert bins_u.shape == u_vals.shape
    assert bins_u.dtype == np.int32
    
    # Test v binning with array
    v_vals = np.array([0.0, 10.0, 20.0, 30.0, 30.0])
    bins_v = grid.bin_v(v_vals)
    assert bins_v.shape == v_vals.shape
    assert bins_v.dtype == np.int32


def test_center_vectorized():
    """Test that center lookup works with arrays."""
    grid = MapGrid(N_u=100, N_v=100, v_max=30.0)
    
    # Test u centers with array
    indices_u = np.array([0, 10, 20, 30, 40])
    centers_u = grid.center_u(indices_u)
    assert centers_u.shape == indices_u.shape
    
    # Test v centers with array
    indices_v = np.array([0, 16, 32, 48, 63])
    centers_v = grid.center_v(indices_v)
    assert centers_v.shape == indices_v.shape

