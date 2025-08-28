# tests/test_io_hdf5.py
import numpy as np
from pathlib import Path

from axion_haloscope.simulation import simulate_spectra
from axion_haloscope.io import SpectrumSet, write_hdf5, read_hdf5

def _to_spectrum_set(spectra, freqs_per_spec, rf_grid, rf_index_map):
    return SpectrumSet(
        spectra=[np.asarray(s, float) for s in spectra],
        freqs_per_spec=[np.asarray(f, float) for f in freqs_per_spec],
        rf_grid=np.asarray(rf_grid, float),
        rf_index_map=[np.asarray(idx, int) for idx in rf_index_map],
    )

def test_hdf5_roundtrip(tmp_path: Path):
    # Small sim for speed
    n_spectra, n_bins, bw = 4, 1000, 100.0
    f_start, step = 5.70e9, 80
    spectra, fper, rf, rf_map = simulate_spectra(
        n_spectra=n_spectra, n_bins=n_bins, bin_width_hz=bw,
        f_start_hz=f_start, tune_step_bins=step, rng_seed=11, axion=None
    )
    sset = _to_spectrum_set(spectra, fper, rf, rf_map)

    # Write & read
    h5file = tmp_path / "spectra.h5"
    write_hdf5(sset, h5file)
    sset2 = read_hdf5(h5file)

    # Equivalence
    assert len(sset2.spectra) == len(sset.spectra)
    assert np.allclose(sset2.rf_grid, sset.rf_grid)

    for a, b in zip(sset.spectra, sset2.spectra):
        assert np.allclose(a, b)

    for a, b in zip(sset.freqs_per_spec, sset2.freqs_per_spec):
        assert np.allclose(a, b)

    for a, b in zip(sset.rf_index_map, sset2.rf_index_map):
        assert np.array_equal(a, b)

