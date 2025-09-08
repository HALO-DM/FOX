# tests/test_data_quality.py
import numpy as np
import pytest
from axion_haloscope.io import SpectrumSet
from axion_haloscope.data_quality import (
    filter_spectrum_set,
    too_noisy,
    identify_bad_spectra,
)

def _toy_sset():
    # Three simple spectra on a common RF grid
    n = 5
    f0 = 5.7e9
    df = 100.0
    freqs = f0 + np.arange(n) * df
    rf_grid = freqs.copy()
    idx = np.arange(n, dtype=int)

    s0 = np.zeros(n) + 1.0                 # clean
    s1 = np.zeros(n) + 1.0; s1[2] = np.nan # has NaN -> should be flagged by too_noisy (nan_fail=True)
    s2 = np.zeros(n) + 5.0                 # large offset but finite

    return SpectrumSet(
        spectra=[s0, s1, s2],
        freqs_per_spec=[freqs, freqs, freqs],
        rf_grid=rf_grid,
        rf_index_map=[idx, idx, idx],
    )

def test_qc_noop_keeps_all():
    sset = _toy_sset()
    filtered, keep, bad = filter_spectrum_set(sset)  # default placeholder -> keep all
    assert keep == [0,1,2]
    assert bad == []
    assert len(filtered.spectra) == 3

def test_qc_predicate_too_noisy_drops_nan_spectrum():
    sset = _toy_sset()
    # Use default too_noisy (nan_fail=True): spectrum 1 has NaN -> dropped
    filtered, keep, bad = filter_spectrum_set(sset, predicate=lambda s,f,i: too_noisy(s,f,i))
    assert bad == [1]
    assert keep == [0,2]
    # rf_index_map stays aligned with kept spectra
    assert np.array_equal(filtered.rf_index_map[0], sset.rf_index_map[0])
    assert np.array_equal(filtered.rf_index_map[1], sset.rf_index_map[2])

def test_qc_explicit_indices_and_mask():
    sset = _toy_sset()
    # explicit indices
    filtered, keep, bad = filter_spectrum_set(sset, bad_indices=[2])
    assert bad == [2] and keep == [0,1] and len(filtered.spectra) == 2
    # boolean mask
    filtered2, keep2, bad2 = filter_spectrum_set(sset, bad_mask=[False, True, False])
    assert bad2 == [1] and keep2 == [0,2]

def test_qc_bad_mask_length_raises():
    sset = _toy_sset()
    with pytest.raises(ValueError):
        filter_spectrum_set(sset, bad_mask=[True, False])  # wrong length
