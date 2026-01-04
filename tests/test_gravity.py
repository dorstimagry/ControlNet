"""Unit tests for gravity compensation."""

import numpy as np
import pytest

from src.maps.gravity import GRAVITY, GravityCompensator, compensate_gravity


def test_gravity_compensator_flat_ground():
    """Test that flat ground (theta=0) doesn't change acceleration."""
    compensator = GravityCompensator()
    
    a_meas = 2.0
    theta = 0.0
    a_dyn = compensator.compensate(a_meas, theta)
    
    assert np.isclose(a_dyn, a_meas)


def test_gravity_compensator_uphill():
    """Test uphill (theta > 0) increases dynamics acceleration."""
    compensator = GravityCompensator()
    
    a_meas = 2.0
    theta = 0.1  # ~5.7 degrees uphill
    a_dyn = compensator.compensate(a_meas, theta)
    
    # Uphill: measured accel is lower than dynamics (gravity opposes motion)
    # So dynamics accel should be higher than measured
    assert a_dyn > a_meas
    expected = a_meas + GRAVITY * np.sin(theta)
    assert np.isclose(a_dyn, expected)


def test_gravity_compensator_downhill():
    """Test downhill (theta < 0) decreases dynamics acceleration."""
    compensator = GravityCompensator()
    
    a_meas = 2.0
    theta = -0.1  # ~5.7 degrees downhill
    a_dyn = compensator.compensate(a_meas, theta)
    
    # Downhill: measured accel is higher than dynamics (gravity aids motion)
    # So dynamics accel should be lower than measured
    assert a_dyn < a_meas
    expected = a_meas + GRAVITY * np.sin(theta)
    assert np.isclose(a_dyn, expected)


def test_gravity_compensator_custom_g():
    """Test with custom gravity value."""
    g_custom = 10.0
    compensator = GravityCompensator(g=g_custom)
    
    a_meas = 2.0
    theta = 0.1
    a_dyn = compensator.compensate(a_meas, theta)
    
    expected = a_meas + g_custom * np.sin(theta)
    assert np.isclose(a_dyn, expected)


def test_gravity_compensator_callable():
    """Test that compensator is callable."""
    compensator = GravityCompensator()
    
    a_meas = 2.0
    theta = 0.1
    
    # Should work as function call
    a_dyn = compensator(a_meas, theta)
    expected = a_meas + GRAVITY * np.sin(theta)
    assert np.isclose(a_dyn, expected)


def test_compensate_gravity_functional():
    """Test functional interface."""
    a_meas = 2.0
    theta = 0.1
    
    a_dyn = compensate_gravity(a_meas, theta)
    expected = a_meas + GRAVITY * np.sin(theta)
    assert np.isclose(a_dyn, expected)


def test_compensate_gravity_vectorized():
    """Test that compensation works with arrays."""
    compensator = GravityCompensator()
    
    a_meas = np.array([1.0, 2.0, 3.0])
    theta = np.array([0.0, 0.1, -0.1])
    
    a_dyn = compensator.compensate(a_meas, theta)
    
    assert a_dyn.shape == a_meas.shape
    expected = a_meas + GRAVITY * np.sin(theta)
    assert np.allclose(a_dyn, expected)


def test_compensate_gravity_broadcast():
    """Test broadcasting with different shapes."""
    a_meas = np.array([1.0, 2.0, 3.0])
    theta = 0.1  # Scalar
    
    a_dyn = compensate_gravity(a_meas, theta)
    
    assert a_dyn.shape == a_meas.shape
    expected = a_meas + GRAVITY * np.sin(theta)
    assert np.allclose(a_dyn, expected)


def test_gravity_sign_convention():
    """Test the sign convention explicitly.
    
    Convention: theta > 0 means nose-up (uphill).
    Gravity contributes: -g*sin(theta) to forward motion.
    Therefore: a_dyn = a_meas + g*sin(theta)
    """
    # Case 1: Going uphill at constant speed
    # Engine must produce more force, so a_dyn > 0 even though a_meas ~ 0
    a_meas = 0.0
    theta = 0.1  # uphill
    a_dyn = compensate_gravity(a_meas, theta)
    assert a_dyn > 0  # Positive dynamics accel needed to maintain speed uphill
    
    # Case 2: Going downhill at constant speed
    # Engine produces less force (or braking), so a_dyn < 0 even though a_meas ~ 0
    a_meas = 0.0
    theta = -0.1  # downhill
    a_dyn = compensate_gravity(a_meas, theta)
    assert a_dyn < 0  # Negative dynamics accel (braking) needed to maintain speed downhill

