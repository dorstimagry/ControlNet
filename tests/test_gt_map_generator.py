"""Unit tests for GT map generator."""

import numpy as np
import pytest

from src.data.gt_map_generator import GTMapGenerator, generate_map_from_params
from src.maps.grid import MapGrid
from utils.dynamics import (
    ExtendedPlantParams,
    ExtendedPlantRandomization,
    sample_extended_params,
)


def test_gt_map_generator_shape():
    """Test that generated map has correct shape."""
    grid = MapGrid(N_u=11, N_v=16, v_max=20.0)
    generator = GTMapGenerator(grid)
    
    # Sample vehicle
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    # Generate map
    X = generator.generate(params, seed=42)
    
    assert X.shape == (11, 16)
    assert X.dtype == np.float32


def test_gt_map_determinism():
    """Test that map generation is deterministic with fixed seed."""
    grid = MapGrid(N_u=11, N_v=16, v_max=20.0)
    generator = GTMapGenerator(grid)
    
    # Sample vehicle
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    # Generate map twice with same seed
    X1 = generator.generate(params, seed=42)
    X2 = generator.generate(params, seed=42)
    
    assert np.allclose(X1, X2)


def test_gt_map_sanity_ranges():
    """Test that map values are in reasonable ranges."""
    grid = MapGrid(N_u=11, N_v=16, v_max=20.0)
    generator = GTMapGenerator(grid)
    
    # Sample vehicle
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    # Generate map
    X = generator.generate(params, seed=42)
    
    # All values should be finite
    assert np.all(np.isfinite(X))
    
    # Not all zeros (that would indicate broken dynamics)
    assert not np.all(X == 0.0)
    
    # Acceleration values should be reasonable
    # Heavy braking can reach -60 m/s², strong throttle can reach 10+ m/s²
    # This is a very loose sanity check
    assert np.all(X > -100.0)  # No ridiculous negative values
    assert np.all(X < 50.0)  # No ridiculous positive values


def test_gt_map_throttle_positive():
    """Test that high throttle generally produces positive acceleration."""
    grid = MapGrid(N_u=11, N_v=16, v_max=20.0)
    generator = GTMapGenerator(grid)
    
    # Sample vehicle
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    # Generate map
    X = generator.generate(params, seed=42)
    
    # At low speeds (first few v bins) with full throttle (last u bin),
    # acceleration should generally be positive
    # u=1.0 is at i_u=-1 (last bin)
    # v=0 is at i_v=0 (first bin)
    accel_full_throttle_low_speed = X[-1, :3]  # Last u bin, first 3 v bins
    
    # Most should be positive (allow some tolerance for edge cases)
    assert np.sum(accel_full_throttle_low_speed > 0) >= 2


def test_gt_map_brake_negative():
    """Test that full brake generally produces negative acceleration."""
    grid = MapGrid(N_u=11, N_v=16, v_max=20.0)
    generator = GTMapGenerator(grid)
    
    # Sample vehicle
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    # Generate map
    X = generator.generate(params, seed=42)
    
    # At moderate speeds with full brake (u=-1.0, first u bin),
    # acceleration should be negative
    accel_full_brake_mod_speed = X[0, 4:8]  # First u bin, middle v bins
    
    # All should be negative
    assert np.all(accel_full_brake_mod_speed < 0)


def test_generate_map_from_params():
    """Test convenience function."""
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    X = generate_map_from_params(
        params,
        N_u=11,
        N_v=16,
        v_max=20.0,
        seed=42,
    )
    
    assert X.shape == (11, 16)
    assert X.dtype == np.float32
    assert np.all(np.isfinite(X))


def test_gt_map_smoothing():
    """Test that smoothing produces different results."""
    grid = MapGrid(N_u=11, N_v=16, v_max=20.0)
    
    # Generate without smoothing
    generator_no_smooth = GTMapGenerator(grid, smoothing_sigma=None)
    
    # Generate with smoothing
    generator_smooth = GTMapGenerator(grid, smoothing_sigma=0.5)
    
    rng = np.random.default_rng(42)
    rand_config = ExtendedPlantRandomization()
    params = sample_extended_params(rng, rand_config)
    
    X_no_smooth = generator_no_smooth.generate(params, seed=42)
    X_smooth = generator_smooth.generate(params, seed=42)
    
    # Should be different
    assert not np.allclose(X_no_smooth, X_smooth)
    
    # But should be correlated (smoothing doesn't destroy structure)
    corr = np.corrcoef(X_no_smooth.flatten(), X_smooth.flatten())[0, 1]
    assert corr > 0.95

