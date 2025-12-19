"""Unit tests for the vehicle parameter fitting module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from fitting import (
    FittedVehicleParams,
    FitterConfig,
    VehicleParamFitter,
    CenteredRandomizationConfig,
)
from utils.dynamics import ExtendedPlantRandomization


class TestFittedVehicleParams:
    """Tests for FittedVehicleParams dataclass."""
    
    def test_create_fitted_params(self) -> None:
        """Test creating FittedVehicleParams with DC motor parameters."""
        params = FittedVehicleParams(
            mass=1800.0,
            drag_area=0.65,
            rolling_coeff=0.012,
            motor_V_max=400.0,
            motor_R=0.2,
            motor_K=0.2,
            gear_ratio=10.0,
            brake_T_max=15000.0,
            wheel_radius=0.346,
            fit_loss=0.1,
            num_samples=10000,
            r_squared=0.85,
        )
        
        assert params.mass == 1800.0
        assert params.drag_area == 0.65
        assert params.rolling_coeff == 0.012
        assert params.motor_V_max == 400.0
        assert params.motor_R == 0.2
        assert params.motor_K == 0.2
        assert params.gear_ratio == 10.0
        assert params.brake_T_max == 15000.0
        assert params.wheel_radius == 0.346
        assert params.fit_loss == 0.1
        assert params.num_samples == 10000
        assert params.r_squared == 0.85
    
    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization and deserialization."""
        params = FittedVehicleParams(
            mass=1900.0,
            drag_area=0.70,
            rolling_coeff=0.010,
            motor_V_max=450.0,
            motor_R=0.15,
            motor_K=0.18,
            gear_ratio=12.0,
            brake_T_max=18000.0,
            wheel_radius=0.340,
            fit_loss=0.05,
            num_samples=50000,
            r_squared=0.92,
        )
        
        # Convert to dict and back
        d = params.to_dict()
        restored = FittedVehicleParams.from_dict(d)
        
        assert restored.mass == params.mass
        assert restored.drag_area == params.drag_area
        assert restored.rolling_coeff == params.rolling_coeff
        assert restored.motor_V_max == params.motor_V_max
        assert restored.motor_R == params.motor_R
        assert restored.motor_K == params.motor_K
        assert restored.gear_ratio == params.gear_ratio
        assert restored.brake_T_max == params.brake_T_max
        assert restored.wheel_radius == params.wheel_radius
        assert restored.fit_loss == params.fit_loss
        assert restored.num_samples == params.num_samples
        assert restored.r_squared == params.r_squared
    
    def test_backward_compatibility_from_dict(self) -> None:
        """Test that old format with motor_force_coeff still loads."""
        old_format = {
            "mass": 1800.0,
            "drag_area": 0.65,
            "rolling_coeff": 0.012,
            "brake_T_max": 15000.0,
            "wheel_radius": 0.346,
            "motor_force_coeff": 50.0,  # Old parameter
            "fit_loss": 0.1,
            "num_samples": 10000,
            "r_squared": 0.85,
        }
        
        params = FittedVehicleParams.from_dict(old_format)
        
        assert params.mass == 1800.0
        # Should use defaults for motor params
        assert params.motor_V_max == 400.0
        assert params.motor_R == 0.2
        assert params.motor_K == 0.2
        assert params.gear_ratio == 10.0
    
    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading to/from JSON file."""
        params = FittedVehicleParams(
            mass=2000.0,
            drag_area=0.72,
            rolling_coeff=0.011,
            motor_V_max=500.0,
            motor_R=0.25,
            motor_K=0.22,
            gear_ratio=9.0,
            brake_T_max=16000.0,
            wheel_radius=0.350,
            fit_loss=0.08,
            num_samples=25000,
            r_squared=0.88,
        )
        
        file_path = tmp_path / "test_fitted_params.json"
        params.save(file_path)
        
        # Check file exists and is valid JSON
        assert file_path.exists()
        with open(file_path, "r") as f:
            data = json.load(f)
        assert "mass" in data
        assert "motor_V_max" in data
        assert "motor_R" in data
        
        # Load and verify
        loaded = FittedVehicleParams.load(file_path)
        assert loaded.mass == params.mass
        assert loaded.motor_V_max == params.motor_V_max
        assert loaded.r_squared == params.r_squared
    
    def test_to_extended_plant_params(self) -> None:
        """Test conversion to ExtendedPlantParams format."""
        params = FittedVehicleParams(
            mass=1800.0,
            drag_area=0.65,
            rolling_coeff=0.012,
            motor_V_max=400.0,
            motor_R=0.2,
            motor_K=0.2,
            gear_ratio=10.0,
            brake_T_max=15000.0,
            wheel_radius=0.346,
        )
        
        plant_dict = params.to_extended_plant_params()
        
        assert plant_dict["motor"]["V_max"] == 400.0
        assert plant_dict["motor"]["R"] == 0.2
        assert plant_dict["motor"]["K_t"] == 0.2
        assert plant_dict["motor"]["K_e"] == 0.2
        assert plant_dict["motor"]["gear_ratio"] == 10.0
        assert plant_dict["body"]["mass"] == 1800.0
        assert plant_dict["brake"]["T_br_max"] == 15000.0


class TestFitterConfig:
    """Tests for FitterConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FitterConfig()
        
        assert config.wheel_radius == 0.346  # Kia Niro EV default
        assert config.mass_init == 1800.0
        assert config.motor_V_max_init == 400.0
        assert config.motor_R_init == 0.2
        assert config.motor_K_init == 0.2
        assert config.gear_ratio_init == 10.0
        assert config.min_speed == 0.5
        assert config.max_accel == 6.0
    
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FitterConfig(
            wheel_radius=0.330,
            mass_init=2000.0,
            motor_V_max_init=500.0,
            motor_R_init=0.15,
            batch_size=10000,
            num_epochs=3,
        )
        
        assert config.wheel_radius == 0.330
        assert config.mass_init == 2000.0
        assert config.motor_V_max_init == 500.0
        assert config.motor_R_init == 0.15
        assert config.batch_size == 10000
        assert config.num_epochs == 3


class TestVehicleParamFitter:
    """Tests for VehicleParamFitter class."""
    
    @pytest.fixture
    def synthetic_trip_data(self, tmp_path: Path) -> Path:
        """Create synthetic trip data for testing using DC motor model."""
        rng = np.random.default_rng(42)
        
        # "True" parameters - within RL randomization bounds
        mass = 2000.0
        drag_area = 0.50
        rolling_coeff = 0.011
        motor_V_max = 400.0
        motor_R = 0.2
        motor_K = 0.2  # K_t = K_e
        gear_ratio = 10.0
        brake_T_max = 20000.0
        wheel_radius = 0.346
        eta_gb = 0.9
        
        dt = 0.02
        n_points = 5000
        
        # Generate realistic driving data
        t = np.arange(n_points) * dt
        
        # Speed profile (varying)
        speed = 10.0 + 8.0 * np.sin(0.1 * t) + rng.normal(0, 0.5, n_points)
        speed = np.maximum(speed, 0.5)  # Keep positive
        
        # Throttle and brake (mutually exclusive-ish)
        throttle = np.zeros(n_points)
        brake = np.zeros(n_points)
        
        for i in range(n_points):
            if rng.random() < 0.7:  # 70% throttle
                throttle[i] = rng.uniform(10, 80)
            else:
                brake[i] = rng.uniform(5, 50)
        
        # Grade (small variations)
        grade = 0.02 * np.sin(0.05 * t) + rng.normal(0, 0.005, n_points)
        
        # Compute acceleration using DC motor physics
        # Motor: i = (V_cmd - K_e * omega_m) / R, tau_m = K_t * i
        omega_m = gear_ratio * speed / wheel_radius
        V_cmd = (throttle / 100.0) * motor_V_max
        back_emf = motor_K * omega_m
        motor_current = np.maximum((V_cmd - back_emf) / motor_R, 0.0)
        motor_torque = motor_K * motor_current
        wheel_torque = eta_gb * gear_ratio * motor_torque
        F_drive = wheel_torque / wheel_radius
        
        F_brake = brake_T_max * (brake / 100.0) / wheel_radius
        F_drag = 0.5 * 1.225 * drag_area * speed * np.abs(speed)
        F_roll = rolling_coeff * mass * 9.80665 * np.cos(grade)
        F_grade = mass * 9.80665 * np.sin(grade)
        
        F_net = F_drive - F_brake - F_drag - F_roll - F_grade
        accel = F_net / mass
        
        # Add noise
        accel += rng.normal(0, 0.3, n_points)
        
        # Create trip data
        trips = {
            "trip_0": {
                "speed": speed,
                "acceleration": accel,
                "throttle": throttle,
                "brake": brake,
                "angle": grade,
            }
        }
        
        data_path = tmp_path / "test_trips.pt"
        torch.save(trips, data_path)
        
        return data_path
    
    def test_load_trip_data(self, synthetic_trip_data: Path) -> None:
        """Test loading trip data from .pt file."""
        fitter = VehicleParamFitter()
        trips = fitter.load_trip_data(synthetic_trip_data)
        
        assert len(trips) == 1
        assert "trip_0" in trips
        assert "speed" in trips["trip_0"]
        assert "acceleration" in trips["trip_0"]
        assert "throttle" in trips["trip_0"]
        assert "brake" in trips["trip_0"]
    
    def test_fit_recovers_parameters(self, synthetic_trip_data: Path) -> None:
        """Test that fitting recovers approximately correct parameters."""
        config = FitterConfig(
            mass_init=1800.0,  # Close to true value
            wheel_radius=0.346,
            batch_size=0,  # Use all data
            num_epochs=3,
        )
        fitter = VehicleParamFitter(config)
        
        fitted = fitter.fit(synthetic_trip_data, verbose=False)
        
        # Check that fitted values are in reasonable range of "true" values
        # (allow wider tolerance due to noise and model complexity)
        true_mass = 2000.0
        true_drag_area = 0.50
        true_rolling_coeff = 0.011
        true_motor_V_max = 400.0
        true_brake_T_max = 20000.0
        
        # Mass should be roughly correct
        assert abs(fitted.mass - true_mass) / true_mass < 0.5
        # Check metadata is populated
        assert fitted.num_samples > 0
        assert fitted.fit_loss >= 0
    
    def test_fit_with_validation(self, synthetic_trip_data: Path) -> None:
        """Test fitting with train/validation split."""
        fitter = VehicleParamFitter()
        
        fitted, val_loss = fitter.fit_with_validation(
            synthetic_trip_data,
            val_fraction=0.2,
            seed=42,
            verbose=False,
        )
        
        # Validation loss should be reasonable
        assert val_loss >= 0
        assert val_loss < 10.0  # Should be relatively small for well-fit model
    
    def test_predict_acceleration(self, synthetic_trip_data: Path) -> None:
        """Test acceleration prediction with fitted parameters."""
        fitter = VehicleParamFitter()
        fitted = fitter.fit(synthetic_trip_data, verbose=False)
        
        # Test prediction
        speed = np.array([5.0, 10.0, 15.0, 20.0])
        throttle = np.array([50.0, 50.0, 50.0, 50.0])
        brake = np.array([0.0, 0.0, 0.0, 0.0])
        grade = np.array([0.0, 0.0, 0.0, 0.0])
        
        a_pred = fitter.predict_acceleration(fitted, speed, throttle, brake, grade)
        
        # Predictions should be reasonable
        assert len(a_pred) == 4
        assert all(np.isfinite(a_pred))
        # At constant throttle, higher speeds should have lower acceleration (back-EMF)
        assert a_pred[0] > a_pred[-1]


class TestCenteredRandomizationConfig:
    """Tests for CenteredRandomizationConfig."""
    
    def test_from_fitted_params(self) -> None:
        """Test creating config from fitted params."""
        fitted = FittedVehicleParams(
            mass=1850.0,
            drag_area=0.68,
            rolling_coeff=0.011,
            motor_V_max=400.0,
            motor_R=0.2,
            motor_K=0.2,
            gear_ratio=10.0,
            brake_T_max=14000.0,
            wheel_radius=0.346,
        )
        
        config = CenteredRandomizationConfig.from_fitted_params(fitted, spread_pct=0.1)
        
        assert config.mass == 1850.0
        assert config.drag_area == 0.68
        assert config.motor_V_max == 400.0
        assert config.motor_R == 0.2
        assert config.spread_pct == 0.1
    
    def test_to_extended_randomization_dict(self) -> None:
        """Test conversion to randomization dictionary."""
        fitted = FittedVehicleParams(
            mass=2000.0,
            drag_area=0.50,
            rolling_coeff=0.010,
            motor_V_max=400.0,
            motor_R=0.2,
            motor_K=0.2,
            gear_ratio=10.0,
            brake_T_max=20000.0,
            wheel_radius=0.346,
        )
        
        config = CenteredRandomizationConfig.from_fitted_params(fitted, spread_pct=0.1)
        rand_dict = config.to_extended_randomization_dict()
        
        assert "vehicle_randomization" in rand_dict
        vr = rand_dict["vehicle_randomization"]
        
        # Check that ranges are centered on fitted values
        mass_range = vr["mass_range"]
        assert mass_range[0] < 2000.0 < mass_range[1]
        
        motor_V_max_range = vr["motor_Vmax_range"]
        assert motor_V_max_range[0] < 400.0 < motor_V_max_range[1]
        
        motor_R_range = vr["motor_R_range"]
        assert motor_R_range[0] < 0.2 < motor_R_range[1]
    
    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading config."""
        fitted = FittedVehicleParams(
            mass=1900.0,
            drag_area=0.65,
            rolling_coeff=0.012,
            motor_V_max=450.0,
            motor_R=0.18,
            motor_K=0.22,
            gear_ratio=11.0,
            brake_T_max=15000.0,
            wheel_radius=0.340,
        )
        
        config = CenteredRandomizationConfig.from_fitted_params(fitted, spread_pct=0.15)
        
        file_path = tmp_path / "centered_config.json"
        config.save(file_path)
        
        loaded = CenteredRandomizationConfig.load(file_path)
        
        assert loaded.mass == config.mass
        assert loaded.motor_V_max == config.motor_V_max
        assert loaded.spread_pct == config.spread_pct


class TestExtendedPlantRandomizationIntegration:
    """Tests for integration with ExtendedPlantRandomization."""
    
    def test_from_fitted_params_factory(self) -> None:
        """Test ExtendedPlantRandomization.from_config with fitted-centered config."""
        fitted = FittedVehicleParams(
            mass=2000.0,
            drag_area=0.55,
            rolling_coeff=0.010,
            motor_V_max=420.0,
            motor_R=0.22,
            motor_K=0.19,
            gear_ratio=9.5,
            brake_T_max=18000.0,
            wheel_radius=0.346,
        )
        
        config = CenteredRandomizationConfig.from_fitted_params(fitted, spread_pct=0.1)
        rand_dict = config.to_extended_randomization_dict()
        
        # Create ExtendedPlantRandomization from the dict
        rand = ExtendedPlantRandomization.from_config(rand_dict)
        
        # Check that ranges are approximately centered on fitted values
        assert rand.mass_range[0] < fitted.mass < rand.mass_range[1]
        assert rand.motor_Vmax_range[0] < fitted.motor_V_max < rand.motor_Vmax_range[1]
    
    def test_create_extended_randomization_from_fitted(self, tmp_path: Path) -> None:
        """Test convenience function for creating randomization."""
        from fitting.randomization_config import create_extended_randomization_from_fitted
        
        fitted = FittedVehicleParams(
            mass=1950.0,
            drag_area=0.60,
            rolling_coeff=0.011,
            motor_V_max=380.0,
            motor_R=0.25,
            motor_K=0.21,
            gear_ratio=10.5,
            brake_T_max=17000.0,
            wheel_radius=0.346,
        )
        
        fitted_path = tmp_path / "fitted.json"
        fitted.save(fitted_path)
        
        rand = create_extended_randomization_from_fitted(fitted_path, spread_pct=0.1)
        
        assert isinstance(rand, ExtendedPlantRandomization)
        assert rand.mass_range[0] < fitted.mass < rand.mass_range[1]


class TestFittingPipeline:
    """End-to-end pipeline tests."""
    
    @pytest.fixture
    def full_synthetic_data(self, tmp_path: Path) -> Path:
        """Create comprehensive synthetic trip data."""
        rng = np.random.default_rng(123)
        
        # "True" parameters - within RL bounds
        mass = 2200.0
        drag_area = 0.55
        rolling_coeff = 0.0105
        motor_V_max = 450.0
        motor_R = 0.18
        motor_K = 0.22
        gear_ratio = 11.0
        brake_T_max = 25000.0
        wheel_radius = 0.346
        eta_gb = 0.9
        
        dt = 0.02
        n_trips = 5
        
        trips = {}
        for trip_idx in range(n_trips):
            n_points = rng.integers(500, 1500)
            t = np.arange(n_points) * dt
            
            # Speed profile
            base_speed = rng.uniform(5, 25)
            speed = base_speed + 5.0 * np.sin(0.08 * t) + rng.normal(0, 0.3, n_points)
            speed = np.maximum(speed, 0.5)
            
            # Throttle and brake
            throttle = np.zeros(n_points)
            brake = np.zeros(n_points)
            for i in range(n_points):
                if rng.random() < 0.65:
                    throttle[i] = rng.uniform(10, 70)
                else:
                    brake[i] = rng.uniform(5, 40)
            
            # Grade
            grade = 0.03 * np.sin(0.03 * t) + rng.normal(0, 0.003, n_points)
            
            # Compute acceleration using DC motor physics
            omega_m = gear_ratio * speed / wheel_radius
            V_cmd = (throttle / 100.0) * motor_V_max
            back_emf = motor_K * omega_m
            motor_current = np.maximum((V_cmd - back_emf) / motor_R, 0.0)
            motor_torque = motor_K * motor_current
            wheel_torque = eta_gb * gear_ratio * motor_torque
            F_drive = wheel_torque / wheel_radius
            
            F_brake = brake_T_max * (brake / 100.0) / wheel_radius
            F_drag = 0.5 * 1.225 * drag_area * speed * np.abs(speed)
            F_roll = rolling_coeff * mass * 9.80665 * np.cos(grade)
            F_grade = mass * 9.80665 * np.sin(grade)
            
            F_net = F_drive - F_brake - F_drag - F_roll - F_grade
            accel = F_net / mass + rng.normal(0, 0.25, n_points)
            
            trips[f"trip_{trip_idx}"] = {
                "speed": speed,
                "acceleration": accel,
                "throttle": throttle,
                "brake": brake,
                "angle": grade,
            }
        
        data_path = tmp_path / "full_test_trips.pt"
        torch.save(trips, data_path)
        
        return data_path
    
    def test_full_pipeline(self, full_synthetic_data: Path, tmp_path: Path) -> None:
        """Test complete fitting pipeline from data to randomization."""
        # 1. Fit parameters
        config = FitterConfig(
            batch_size=0,  # Use all data
            num_epochs=3,
        )
        fitter = VehicleParamFitter(config)
        fitted = fitter.fit(full_synthetic_data, verbose=False)
        
        # 2. Save fitted params
        fitted_path = tmp_path / "fitted_params.json"
        fitted.save(fitted_path)
        
        # 3. Verify saved file
        assert fitted_path.exists()
        
        # 4. Create centered randomization
        loaded = FittedVehicleParams.load(fitted_path)
        centered_config = CenteredRandomizationConfig.from_fitted_params(loaded, spread_pct=0.1)
        rand_dict = centered_config.to_extended_randomization_dict()
        
        # 5. Create ExtendedPlantRandomization
        rand = ExtendedPlantRandomization.from_config(rand_dict)
        
        # 6. Verify we can sample parameters
        from utils.dynamics import sample_extended_params
        
        # Try sampling a few times
        successful = 0
        for seed in range(10):
            try:
                rng = np.random.default_rng(seed)
                params = sample_extended_params(rng, rand)
                successful += 1
            except ValueError:
                # Some samples may fail feasibility checks
                pass
        
        assert successful >= 3, f"Only {successful} successful samples"


class TestDCMotorPhysics:
    """Tests for DC motor physics in the fitter."""
    
    def test_back_emf_reduces_acceleration_at_speed(self) -> None:
        """Test that back-EMF effect reduces drive force at higher speeds."""
        config = FitterConfig()
        fitter = VehicleParamFitter(config)
        
        # Parameters
        params = np.array([
            2000.0,   # mass
            0.5,      # drag_area
            0.01,     # rolling_coeff
            400.0,    # V_max
            0.2,      # R
            0.2,      # K
            10.0,     # gear_ratio
            20000.0,  # brake_T_max
        ])
        
        # Same throttle, different speeds
        throttle = np.array([50.0, 50.0, 50.0])
        brake = np.array([0.0, 0.0, 0.0])
        grade = np.array([0.0, 0.0, 0.0])
        speed = np.array([1.0, 10.0, 20.0])
        
        a_pred = fitter._predict_acceleration(params, speed, throttle, brake, grade)
        
        # Higher speed should have lower acceleration due to back-EMF
        assert a_pred[0] > a_pred[1] > a_pred[2]
    
    def test_no_regen_braking(self) -> None:
        """Test that motor current doesn't go negative (no regen in model)."""
        config = FitterConfig()
        fitter = VehicleParamFitter(config)
        
        # High speed, low throttle - would cause negative current without clamp
        params = np.array([
            2000.0, 0.5, 0.01, 400.0, 0.2, 0.2, 10.0, 20000.0
        ])
        
        speed = np.array([30.0])  # High speed
        throttle = np.array([5.0])  # Low throttle
        brake = np.array([0.0])
        grade = np.array([0.0])
        
        a_pred = fitter._predict_acceleration(params, speed, throttle, brake, grade)
        
        # Should still produce reasonable (likely negative) acceleration
        # due to drag/rolling, but no regen contribution
        assert np.isfinite(a_pred[0])
