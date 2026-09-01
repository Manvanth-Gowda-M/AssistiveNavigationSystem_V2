"""
Unit Tests - Configuration Loader
===================================
Phase 1 PASS condition: config loads without error.

Run with:
    .venv/Scripts/python.exe -m pytest tests/unit/test_config_loader.py -v
"""

import pytest
from pathlib import Path


def test_config_loads_successfully():
    """config/config.yaml must load without raising any exception."""
    from assistive_navigation.utils.config_loader import load_config
    config = load_config()
    assert config is not None, "Config should not be None"
    assert isinstance(config, dict), "Config should be a dictionary"


def test_config_has_required_sections():
    """All top-level sections must be present."""
    from assistive_navigation.utils.config_loader import load_config
    config = load_config()
    required_sections = [
        "camera", "detection", "object_classes", "tracking",
        "depth", "proximity", "spatial", "temporal", "priority",
        "alerts", "audio", "debug", "logging"
    ]
    for section in required_sections:
        assert section in config, f"Missing config section: '{section}'"


def test_camera_config():
    """Camera config values must be present and valid."""
    from assistive_navigation.utils.config_loader import load_config
    config = load_config()
    cam = config["camera"]
    assert isinstance(cam["device_id"], int), "device_id must be an integer"
    assert cam["width"] > 0, "width must be positive"
    assert cam["height"] > 0, "height must be positive"
    assert cam["max_fps"] > 0, "max_fps must be positive"


def test_detection_config():
    """Detection config values must be valid."""
    from assistive_navigation.utils.config_loader import load_config
    config = load_config()
    det = config["detection"]
    assert 0.0 < det["confidence_threshold"] < 1.0, \
        "confidence_threshold must be between 0 and 1"
    assert det["input_size"] > 0, "input_size must be positive"
    assert det["device"] in ("cpu", "cuda"), \
        "device must be 'cpu' or 'cuda'"


def test_missing_config_raises_error(tmp_path):
    """Loading a non-existent config file must raise FileNotFoundError."""
    from assistive_navigation.utils.config_loader import load_config
    with pytest.raises(FileNotFoundError):
        load_config(config_path=tmp_path / "nonexistent.yaml")


def test_project_root_exists():
    """Project root directory must exist."""
    from assistive_navigation.utils.config_loader import get_project_root
    root = get_project_root()
    assert root.exists(), f"Project root not found: {root}"
    assert root.is_dir(), "Project root must be a directory"


def test_models_dir_exists():
    """Models directory must exist."""
    from assistive_navigation.utils.config_loader import get_models_dir
    models_dir = get_models_dir()
    assert models_dir.exists(), f"Models directory not found: {models_dir}"


def test_logs_dir_exists():
    """Logs directory must exist."""
    from assistive_navigation.utils.config_loader import get_logs_dir
    logs_dir = get_logs_dir()
    assert logs_dir.exists(), f"Logs directory not found: {logs_dir}"
