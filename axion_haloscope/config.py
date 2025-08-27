# axion_haloscope/config.py
"""
Global configuration for output paths.
"""

import pathlib
import os

# Default output directory: ./output (relative to where you run Python)
OUTPUT_DIR = pathlib.Path(os.environ.get("AXION_HALO_OUTPUT", "./output"))

def ensure_output_dir() -> pathlib.Path:
    """Create OUTPUT_DIR if needed and return it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
