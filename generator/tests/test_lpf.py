"""Tests for the second-order LPF implementation."""

import torch
import pytest

from generator.lpf import SecondOrderLPF, second_order_lpf_update


class TestSecondOrderLPFUpdate:
    """Test the core LPF update function."""

    def test_basic_update(self):
        """Test basic LPF update with step input."""
        device = torch.device('cpu')

        # Initial conditions
        y = torch.tensor([0.0], device=device)
        r = torch.tensor([0.0], device=device)
        u = torch.tensor([1.0], device=device)
        dt = 0.02
        wc = 2 * torch.pi * 0.6  # 0.6 Hz cutoff
        zeta = 0.9
        rate_max = 4.0
        rate_neg_max = 8.0
        jerk_max = 12.0

        # Update
        y_new, r_new = second_order_lpf_update(
            y, r, u, dt, wc, zeta, rate_max, rate_neg_max, jerk_max, device
        )

        # Basic checks
        assert y_new.shape == y.shape
        assert r_new.shape == r.shape
        assert y_new.item() > 0  # Should start moving toward target
        assert abs(r_new.item()) <= rate_max  # Rate should be limited

    def test_rate_limits(self):
        """Test that rate limits are enforced."""
        device = torch.device('cpu')

        # Large step to test rate limiting
        y = torch.tensor([0.0], device=device)
        r = torch.tensor([0.0], device=device)
        u = torch.tensor([10.0], device=device)  # Large target
        dt = 0.02
        wc = 2 * torch.pi * 0.6
        zeta = 0.9
        rate_max = 2.0  # Tight limit
        rate_neg_max = 2.0
        jerk_max = 12.0

        y_new, r_new = second_order_lpf_update(
            y, r, u, dt, wc, zeta, rate_max, rate_neg_max, jerk_max, device
        )

        assert abs(r_new.item()) <= rate_max + 1e-6

    def test_vectorized_update(self):
        """Test vectorized operation across batch."""
        device = torch.device('cpu')
        batch_size = 3

        y = torch.zeros(batch_size, device=device)
        r = torch.zeros(batch_size, device=device)
        u = torch.tensor([0.0, 1.0, 2.0], device=device)
        dt = 0.02
        wc = 2 * torch.pi * 0.6
        zeta = 0.9
        rate_max = 4.0
        rate_neg_max = 8.0
        jerk_max = 12.0

        y_new, r_new = second_order_lpf_update(
            y, r, u, dt, wc, zeta, rate_max, rate_neg_max, jerk_max, device
        )

        assert y_new.shape == (batch_size,)
        assert r_new.shape == (batch_size,)
        assert torch.all(y_new >= 0)  # Should move toward respective targets


class TestSecondOrderLPF:
    """Test the LPF class."""

    def test_initialization(self):
        """Test LPF class initialization."""
        batch_size = 4
        lpf = SecondOrderLPF(batch_size=batch_size, freq_cutoff=0.6)

        assert lpf.batch_size == batch_size
        assert lpf.y.shape == (batch_size,)
        assert lpf.r.shape == (batch_size,)
        assert torch.all(lpf.y == 0)
        assert torch.all(lpf.r == 0)

    def test_step_response(self):
        """Test step response behavior."""
        lpf = SecondOrderLPF(batch_size=1, freq_cutoff=0.6, dt=0.02)
        u = torch.tensor([1.0])

        # Take a few steps
        for _ in range(10):
            y = lpf.update(u)

        # Should approach target but not reach instantly
        assert y.item() > 0
        assert y.item() < 1.0

    def test_reset(self):
        """Test reset functionality."""
        lpf = SecondOrderLPF(batch_size=2, freq_cutoff=0.6)

        # Update a bit
        u = torch.tensor([1.0, 2.0])
        lpf.update(u)
        assert not torch.all(lpf.y == 0)

        # Reset
        lpf.reset()
        assert torch.all(lpf.y == 0)
        assert torch.all(lpf.r == 0)

    def test_custom_initial_conditions(self):
        """Test custom initial conditions."""
        initial_y = torch.tensor([1.0, 2.0])
        initial_r = torch.tensor([0.1, -0.1])

        lpf = SecondOrderLPF(batch_size=2, freq_cutoff=0.6)
        lpf.reset(initial_y, initial_r)

        assert torch.allclose(lpf.y, initial_y)
        assert torch.allclose(lpf.r, initial_r)
