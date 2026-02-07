"""Unit tests for speed shaper configuration schema."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from speed_shaper.src.config_schema import ShaperConfig


def test_shaper_config_creation():
    """Test basic creation of ShaperConfig."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
    )
    
    assert config.dt == 0.1
    assert config.wE_start == 30.0
    assert config.wA_end == 10.0
    assert config.lamJ == -0.5
    assert config.enable_accel_bounds is False
    assert config.a_min is None


def test_shaper_config_with_bounds():
    """Test ShaperConfig with acceleration and jerk bounds."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
        a_min=-3.0,
        a_max=3.0,
        j_min=-8.0,
        j_max=8.0,
        enable_accel_bounds=True,
        enable_jerk_bounds=True,
    )
    
    assert config.enable_accel_bounds is True
    assert config.enable_jerk_bounds is True
    assert config.a_min == -3.0
    assert config.a_max == 3.0
    assert config.j_min == -8.0
    assert config.j_max == 8.0


def test_shaper_config_to_dict():
    """Test conversion to dictionary."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
        metadata={'test': 'value'}
    )
    
    data = config.to_dict()
    assert isinstance(data, dict)
    assert data['dt'] == 0.1
    assert data['wE_start'] == 30.0
    assert data['lamA'] == -0.5
    assert data['metadata'] == {'test': 'value'}


def test_shaper_config_from_dict():
    """Test creation from dictionary."""
    data = {
        'dt': 0.1,
        'wE_start': 30.0,
        'wE_end': 10.0,
        'lamE': 1.0,
        'wA_start': 2.0,
        'wA_end': 10.0,
        'lamA': -0.5,
        'wJ_start': 2.0,
        'wJ_end': 10.0,
        'lamJ': -0.5,
        'a_min': -3.0,
        'a_max': 3.0,
        'enable_accel_bounds': True,
        'metadata': {'source': 'test'}
    }
    
    config = ShaperConfig.from_dict(data)
    assert config.dt == 0.1
    assert config.wE_start == 30.0
    assert config.lamA == -0.5
    assert config.a_min == -3.0
    assert config.enable_accel_bounds is True
    assert config.metadata == {'source': 'test'}


def test_shaper_config_from_dict_missing_required():
    """Test that missing required fields raise ValueError."""
    data = {
        'dt': 0.1,
        'wE_start': 30.0,
        # Missing other required fields
    }
    
    with pytest.raises(ValueError, match="Missing required fields"):
        ShaperConfig.from_dict(data)


def test_shaper_config_json_roundtrip(tmp_path):
    """Test saving and loading JSON."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
        a_min=-3.0,
        a_max=3.0,
        enable_accel_bounds=True,
        metadata={'test': 'roundtrip'}
    )
    
    # Save
    filepath = tmp_path / "test_config.json"
    config.to_json(filepath)
    
    assert filepath.exists()
    
    # Load
    loaded = ShaperConfig.from_json(filepath)
    assert loaded.dt == config.dt
    assert loaded.wE_start == config.wE_start
    assert loaded.lamA == config.lamA
    assert loaded.a_min == config.a_min
    assert loaded.enable_accel_bounds == config.enable_accel_bounds
    assert 'test' in loaded.metadata


def test_shaper_config_json_adds_timestamp(tmp_path):
    """Test that JSON save adds timestamp if not present."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
    )
    
    filepath = tmp_path / "test_config_timestamp.json"
    config.to_json(filepath)
    
    # Read JSON and check timestamp was added
    with filepath.open('r') as f:
        data = json.load(f)
    
    assert 'metadata' in data
    assert 'created_at' in data['metadata']
    # Check it's a valid ISO format timestamp
    datetime.fromisoformat(data['metadata']['created_at'])


def test_shaper_config_from_json_nonexistent():
    """Test that loading nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ShaperConfig.from_json(Path("/nonexistent/path.json"))


def test_shaper_config_with_optional_none_values():
    """Test that None values for bounds are handled correctly."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
        a_min=None,
        a_max=None,
        j_min=None,
        j_max=None,
        enable_accel_bounds=False,
    )
    
    data = config.to_dict()
    assert data['a_min'] is None
    assert data['a_max'] is None
    
    # Roundtrip
    loaded = ShaperConfig.from_dict(data)
    assert loaded.a_min is None
    assert loaded.a_max is None


def test_shaper_config_str():
    """Test string representation."""
    config = ShaperConfig(
        dt=0.1,
        wE_start=30.0,
        wE_end=10.0,
        lamE=1.0,
        wA_start=2.0,
        wA_end=10.0,
        lamA=-0.5,
        wJ_start=2.0,
        wJ_end=10.0,
        lamJ=-0.5,
        a_min=-3.0,
        a_max=3.0,
        enable_accel_bounds=True,
    )
    
    s = str(config)
    assert 'ShaperConfig' in s
    assert '0.1s' in s
    assert 'Error weights' in s
    assert 'Accel bounds' in s
    assert '-3.0' in s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
