"""
Sets
====

Module to create sets of spectra and supporting information 
"""

from typing import List, Tuple, Dict
import numpy as np

from axion_haloscope.baseline import remove_baseline
from axion_haloscope.io_working import SpectrumMetadata

def group_sets(
    dts: List,
    spacing_minutes: float,
    specs: np.ndarray,
    fper: np.ndarray,
    metadata: SpectrumMetadata
) -> List[List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Group spectra into "sets" (observation runs) by
    grouping timestamps that are separated by less than `spacing_minutes`.
    A new set starts whenever the gap to the next timestamp is >= threshold.

    Parameters
    ==========
    dts: list of datetime
        Timestamp of each observation, assumed sorted ascending
    spacing_minutes: float
        Maximum gap (in minutes) between consecutive observations for them
        to be considered part of the same set
    specs: list of np.ndarray
        Spectrum for each observation, aligned with `dts`
    fper: list of np.ndarray
        Frequency-per-bin axis for each observation, aligned with `dts`
    metadata: object
        Must expose `res_freqs`, a sequence of resonant frequencies aligned
        with `dts`

    Returns
    =======
    sets: list of list of tuple
        One list per observation set; each tuple is
        (spectrum, freq_per_bin, resonant_freq) for one observation in
        that set
    """
    sets = []
    n = len(dts)
    threshold = spacing_minutes * 60  # seconds
    i = 0
    while i < n:
        j = i + 1
        while j < n and (dts[j] - dts[i]).total_seconds() < threshold:
            j += 1
        sets.append([(specs[k], fper[k], metadata.res_freqs[k]) for k in range(i, j)])
        i = j
    return sets

def set_averaging(
    sets: List[List[Tuple[np.ndarray, np.ndarray, float]]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Average the spectra and frequency axes within each observation set.

    Parameters
    ==========
    sets: list of list of tuple
        Output of `group_sets`: each element is a list of
        (spectrum, freq_per_bin, resonant_freq) tuples for one set

    Returns
    =======
    set_avg_spectra: list of tuple
        One (freq_per_bin_avg, spectrum_avg) tuple per set, each averaged
        across all observations in that set
    """
    set_avg_spectra = []
    for single_set in sets:
        set_avg_spectra.append(None)

        set_avg_spectra.append((np.mean([x[1] for x in single_set], axis=0),
                                np.mean([x[0] for x in single_set], axis=0)))
    return set_avg_spectra

def set_average_baseline_fitting(
    set_avg_spectra: List[Tuple[np.ndarray, np.ndarray]],
    base: Dict,
) -> List[np.ndarray]:
    """
    Fit a Savitzky-Golay baseline to each set's averaged spectrum.

    Parameters
    ==========
    set_avg_spectra: list of tuple
        Output of `set_averaging`: (freq_per_bin_avg, spectrum_avg) per set
    base: dict
        Baseline-fitting config; must contain "sg_window_warm" and
        "sg_poly_warm" (Savitzky-Golay window length and polynomial order)

    Returns
    =======
    set_sg_fits: list of np.ndarray or None
        Fitted baseline per set, aligned with `set_avg_spectra`. `None`
        where the averaged spectrum was degenerate (all-zero/empty).
    """
    set_sg_fits = []
    for _, spec_avg in set_avg_spectra:
        if not spec_avg.any():
            set_sg_fits.append(None)
            continue

        _, baseline = remove_baseline(
                spectrum=spec_avg,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
                )
        set_sg_fits.append(baseline)
    return set_sg_fits

def set_creation(
    dts: List,
    spacing_minutes: float,
    specs: np.ndarray,
    fper: np.ndarray,
    metadata: SpectrumMetadata,
    base: Dict
) -> Tuple:
    """
    Group observations into sets, average each set's spectra, and fit a
    baseline to each set's averaged spectrum.

    Parameters
    ==========
    dts, spacing_minutes, specs, fper, metadata: see `group_sets`
    base: see `set_average_baseline_fitting`

    Returns
    =======
    sets: list of list of tuple
        See `group_sets`
    set_avg_spectra: list of tuple
        See `set_averaging`
    set_sg_fits: list of np.ndarray or None
        See `set_average_baseline_fitting`
    """
    sets = group_sets(dts, spacing_minutes, specs, fper, metadata)
    set_avg_spectra = set_averaging(sets)
    set_sg_fits = set_average_baseline_fitting(set_avg_spectra, base)
    return sets, set_avg_spectra, set_sg_fits
