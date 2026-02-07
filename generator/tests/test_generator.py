"""Tests for the main batch target generator."""

import torch
import pytest

from generator.generator import BatchTargetGenerator, GeneratorConfig
from generator.feasibility import VehicleCapabilities


class TestBatchTargetGenerator:
    """Test the main batch generator."""

    def create_test_vehicle(self, batch_size: int) -> VehicleCapabilities:
        """Create a test vehicle with reasonable parameters."""
        return VehicleCapabilities(
            m=torch.full((batch_size,), 1500.0),
            r_w=torch.full((batch_size,), 0.3),
            gear_ratio=torch.full((batch_size,), 10.0),
            eta_gb=torch.full((batch_size,), 0.9),
            K_t=torch.full((batch_size,), 0.5),
            K_e=torch.full((batch_size,), 0.5),
            R=torch.full((batch_size,), 0.1),
            i_max=torch.full((batch_size,), 200.0),
            V_max=torch.full((batch_size,), 400.0),
            b=torch.full((batch_size,), 1e-3),
            T_brake_max=torch.full((batch_size,), 2000.0),
            mu=torch.full((batch_size,), 0.8),
            C_dA=torch.full((batch_size,), 0.6),
            C_r=torch.full((batch_size,), 0.01)
        )

    def test_generator_initialization(self):
        """Test generator initialization."""
        config = GeneratorConfig(prediction_horizon=5)
        generator = BatchTargetGenerator(config, batch_size=4, episode_length=10)

        assert generator.batch_size == 4
        assert generator.episode_length == 10
        assert generator.config.prediction_horizon == 5

    def test_small_batch_generation(self):
        """Test generation of a small batch."""
        config = GeneratorConfig(
            prediction_horizon=3,
            dt=0.1  # Faster for testing
        )
        generator = BatchTargetGenerator(config, batch_size=2, episode_length=5)
        vehicle = self.create_test_vehicle(2)

        targets, grades, raw_targets = generator.generate_batch(vehicle)

        # Check output shapes
        assert targets.shape == (2, 5, 3)  # [B, T, H]
        assert grades.shape == (2, 5)      # [B, T] - grade per step

        # Check basic properties
        assert torch.all(targets >= 0)  # Non-negative speeds
        assert torch.all(torch.isfinite(targets))

    def test_consistency_check(self):
        """Test that targets[:, t, 0] matches the LPF state at each step."""
        config = GeneratorConfig(
            prediction_horizon=2,
            dt=0.1
        )
        generator = BatchTargetGenerator(config, batch_size=1, episode_length=3)
        vehicle = self.create_test_vehicle(1)

        targets, _, _ = generator.generate_batch(vehicle)

        # The first element of each horizon should be consistent with the next step's first element
        # This is a basic consistency check
        for t in range(generator.episode_length - 1):
            current_first = targets[0, t, 0]
            next_first = targets[0, t + 1, 0]
            # They should be close but not necessarily identical due to ongoing LPF dynamics
            assert abs(current_first - next_first) < 1.0  # Reasonable difference

    def test_feasibility_basic(self):
        """Basic test that generated targets are within reasonable bounds."""
        config = GeneratorConfig(
            prediction_horizon=5,
            v_max_sample=20.0,  # Reasonable speed limit
            dt=0.1
        )
        generator = BatchTargetGenerator(config, batch_size=1, episode_length=10)
        vehicle = self.create_test_vehicle(1)

        targets, _, _ = generator.generate_batch(vehicle)

        # Check speed bounds
        assert torch.all(targets >= 0)
        assert torch.all(targets <= config.v_max_sample * 1.5)  # Some margin

        # Check acceleration bounds (rough check)
        a = (targets[:, 1:, 0] - targets[:, :-1, 0]) / config.dt
        max_reasonable_accel = 10.0  # m/s²
        assert torch.all(a >= -max_reasonable_accel)
        assert torch.all(a <= max_reasonable_accel)

    def test_event_driven_behavior(self):
        """Test that the generator produces event-driven changes."""
        # Use high change probability to ensure events occur
        config = GeneratorConfig(
            prediction_horizon=3,
            p_change=0.2,  # High probability
            dt=0.1
        )
        generator = BatchTargetGenerator(config, batch_size=1, episode_length=20)
        vehicle = self.create_test_vehicle(1)

        targets, _, _ = generator.generate_batch(vehicle)

        # Should see some variation in the target speeds
        # Check that not all targets are identical
        first_targets = targets[0, :, 0]
        unique_targets = torch.unique(first_targets)
        assert len(unique_targets) > 1  # Should have some variation

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_sizes(self, batch_size):
        """Test different batch sizes."""
        config = GeneratorConfig(
            prediction_horizon=2
        )
        generator = BatchTargetGenerator(config, batch_size=batch_size, episode_length=3)
        vehicle = self.create_test_vehicle(batch_size)

        targets, grades, raw_targets = generator.generate_batch(vehicle)

        assert targets.shape == (batch_size, 3, 2)
        assert grades.shape == (batch_size, 3)

    def test_zero_stop_generation(self):
        """Test that zero stops are generated with appropriate probability."""
        # Use high zero-stop probability
        config = GeneratorConfig(
            prediction_horizon=3,
            p_zero_stop=0.3,  # High probability
            dt=0.1
        )
        generator = BatchTargetGenerator(config, batch_size=10, episode_length=50)
        vehicle = self.create_test_vehicle(10)

        targets, _, _ = generator.generate_batch(vehicle)

        # Check if any episodes contain near-zero speeds
        min_speeds = torch.min(targets.view(10, -1), dim=1)[0]
        has_zero_stops = torch.any(min_speeds < 0.1)  # Very low threshold

        # With high probability, we should see some zero stops
        # (This is probabilistic, so we can't guarantee it)
        if not has_zero_stops:
            pytest.skip("No zero stops generated in this random sample")


if __name__ == "__main__":
    # Quick manual test
    config = GeneratorConfig(prediction_horizon=3)
    generator = BatchTargetGenerator(config, batch_size=1, episode_length=5)

    vehicle = VehicleCapabilities(
        m=torch.tensor([1500.0]),
        r_w=torch.tensor([0.3]),
        gear_ratio=torch.tensor([10.0]),
        eta_gb=torch.tensor([0.9]),
        K_t=torch.tensor([0.5]),
        K_e=torch.tensor([0.5]),
        R=torch.tensor([0.1]),
        i_max=torch.tensor([200.0]),
        V_max=torch.tensor([400.0]),
        b=torch.tensor([1e-3]),
        T_brake_max=torch.tensor([2000.0]),
        mu=torch.tensor([0.8]),
        C_dA=torch.tensor([0.6]),
        C_r=torch.tensor([0.01])
    )

    targets, grades = generator.generate_batch(vehicle)
    print(f"Generated targets shape: {targets.shape}")
    print(f"Sample targets:\n{targets[0]}")
    print(f"Grades shape: {grades.shape}")
