# axion_haloscope/__init__.py
"""
Axion Haloscope Analysis (toy HAYSTAC-like pipeline)

Convenience exports for the main analysis steps:
- simulation
- baseline removal
- vertical combination
- rebinning & matched filter
- candidate search
- exclusion limit
"""

__version__ = "0.1.0"

# Re-export core APIs
from .simulation import simulate_spectra, AxionParams
from .baseline import remove_baseline, mask_bins
from .combine import combine_ml
from .rebin import rebin_ml, axion_template_gaussian, grand_spectrum_ml
from .detection import threshold_for_detection, find_candidates
from .limit import compute_local_snr_template, coupling_limit, plot_exclusion
from .lineshape import shm_maxwell_template, shm_maxwell_profile
from .io import write_hdf5, read_hdf5


__all__ = [
    "simulate_spectra", "AxionParams",
    "remove_baseline", "mask_bins",
    "combine_ml",
    "rebin_ml", "axion_template_gaussian", "grand_spectrum_ml",
    "threshold_for_detection", "find_candidates",
    "compute_local_snr_template", "coupling_limit", "plot_exclusion",
    "shm_maxwell_template", "shm_maxwell_profile",
    "write_hdf5", "read_hdf5",
]
