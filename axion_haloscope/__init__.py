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
from .limit import compute_local_snr_template, coupling_limit
from .lineshape import shm_maxwell_template, shm_maxwell_profile

from .io_working import write_hdf5, read_hdf5,read_qshs_hdf5, read_qshs_hdf5_dir
from .data_quality_working import filter_spectrum_set, identify_bad_spectra, placeholder_bad_predicate



__all__ = [
    "simulate_spectra", "AxionParams",
    "remove_baseline", "mask_bins",
    "combine_ml",
    "rebin_ml", "axion_template_gaussian", "grand_spectrum_ml",
    "threshold_for_detection", "find_candidates",
    "compute_local_snr_template", "coupling_limit",
    "shm_maxwell_template", "shm_maxwell_profile",
    "write_hdf5", "read_hdf5",    "read_qshs_hdf5", "read_qshs_hdf5_dir",
    "filter_spectrum_set", "identify_bad_spectra", "placeholder_bad_predicate","too_noisy"
]



from .data_quality_working import restrict_frequency_range
__all__ += ["restrict_frequency_range"]
