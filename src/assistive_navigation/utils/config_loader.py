"""
Configuration Loader
=====================
Loads and validates config/config.yaml.

This is one of the first modules to be fully implemented because every
other module depends on it.

Usage:
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    threshold = config['detection']['confidence_threshold']

The config file path is resolved relative to the project root.
If the config file is missing, a clear error message is shown — not a
cryptic KeyError or FileNotFoundError.
"""

import os
import yaml
from pathlib import Path


# Resolve project root: two levels up from this file
# (this file is at src/assistive_navigation/utils/config_loader.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load and return the configuration dictionary from config.yaml.

    Args:
        config_path: Optional path to config file. Defaults to
                     config/config.yaml in the project root.

    Returns:
        dict: The full configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file has invalid YAML syntax.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {_DEFAULT_CONFIG_PATH}\n"
            f"Make sure you are running from the project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse configuration file: {config_path}\n"
                f"YAML error: {e}"
            ) from e

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def get_models_dir() -> Path:
    """Return the absolute path to the models directory."""
    return _PROJECT_ROOT / "models"


def get_logs_dir() -> Path:
    """Return the absolute path to the logs directory."""
    return _PROJECT_ROOT / "logs"


if __name__ == "__main__":
    # Quick self-test: run this file directly to verify config loads
    print(f"Project root: {get_project_root()}")
    print(f"Config path:  {_DEFAULT_CONFIG_PATH}")
    cfg = load_config()
    print(f"Config loaded successfully.")
    print(f"  Camera device:      {cfg['camera']['device_id']}")
    print(f"  Detection model:    {cfg['detection']['model_name']}")
    print(f"  Confidence threshold: {cfg['detection']['confidence_threshold']}")
    print(f"  TTS engine:         {cfg['audio']['engine']}")
    print(f"  Debug enabled:      {cfg['debug']['enabled']}")
    print("Config loader: OK")
