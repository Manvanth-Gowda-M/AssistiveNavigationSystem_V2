"""
pytest configuration and shared fixtures.

This file is automatically loaded by pytest before any test runs.
Add shared fixtures here that multiple test files need.
"""

import sys
from pathlib import Path

# Ensure src/ is on the Python path so tests can import our package
# without needing the package to be pip-installed.
_project_root = Path(__file__).resolve().parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
