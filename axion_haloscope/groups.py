"""
Groups
====

Module to create groups of spectra and supporting information 
"""

from typing import List, Tuple, Dict
import numpy as np

from axion_haloscope.baseline import remove_baseline
from axion_haloscope.io_working import SpectrumMetadata

def group_spectra(
    dts: List,
    spacing_minutes: float,
    specs: np.ndarray,
    fper: np.ndarray,
    metadata: SpectrumMetadata
) -> List[List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    Group spectra into "groups" (observation runs) by
    grouping timestamps that are separated by less than `spacing_minutes`.
    A new group starts whenever the gap to the next timestamp is >= threshold.

    Parameters
    ==========
    dts: list of datetime
        Timestamp of each observation, assumed sorted ascending
    spacing_minutes: float
        Maximum gap (in minutes) between consecutive observations for them
        to be considered part of the same group
    specs: list of np.ndarray
        Spectrum for each observation, aligned with `dts`
    fper: list of np.ndarray
        Frequency-per-bin axis for each observation, aligned with `dts`
    metadata: object
        Must expose `res_freqs`, a sequence of resonant frequencies aligned
        with `dts`

    Returns
    =======
    grand_group: list of list of tuple
        A grand group containing all groups.
        One list per observation group; each tuple is
        (spectrum, freq_per_bin, resonant_freq) for one observation in
        that group.
    """
    grand_group = []
    n = len(dts)
    threshold = spacing_minutes * 60  # seconds
    i = 0
    while i < n:
        j = i + 1
        while j < n and (dts[j] - dts[i]).total_seconds() < threshold:
            j += 1
        grand_group.append([(specs[k], fper[k], metadata.res_freqs[k]) for k in range(i, j)])
        i = j
    return grand_group

def group_averaging(
    grand_group: List[List[Tuple[np.ndarray, np.ndarray, float]]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Average the spectra and frequency axes within each observation group.

    Parameters
    ==========
    grand_group: list of list of tuple
        Output of `group_spectra`: each element is a list of
        (spectrum, freq_per_bin, resonant_freq) tuples for one group

    Returns
    =======
    group_avg_spectra: list of tuple
        One (freq_per_bin_avg, spectrum_avg) tuple per group, each averaged
        across all observations in that group
    """
    group_avg_spectra = []
    for group in grand_group:
        if group is None:
            group_avg_spectra.append(None)
            continue

        group_avg_spectra.append((np.mean([x[1] for x in group], axis=0),
                                np.mean([x[0] for x in group], axis=0)))
    return group_avg_spectra

def group_average_baseline_fitting(
    group_avg_spectra: List[Tuple[np.ndarray, np.ndarray]],
    base: Dict,
) -> List[np.ndarray]:
    """
    Fit a Savitzky-Golay baseline to each group's averaged spectrum.

    Parameters
    ==========
    group_avg_spectra: list of tuple
        Output of `group_averaging`: (freq_per_bin_avg, spectrum_avg) per group
    base: dict
        Baseline-fitting config; must contain "sg_window_warm" and
        "sg_poly_warm" (Savitzky-Golay window length and polynomial order)

    Returns
    =======
    group_sg_fits: list of np.ndarray or None
        Fitted baseline per group, aligned with `group_avg_spectra`. `None`
        where the averaged spectrum was degenerate (all-zero/empty).
    """
    group_sg_fits = []
    for _, spec_avg in group_avg_spectra:
        if not spec_avg.any():
            group_sg_fits.append(None)
            continue

        _, baseline = remove_baseline(
                spectrum=spec_avg,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
                )
        group_sg_fits.append(baseline)
    return group_sg_fits

def group_creation(
    dts: List,
    spacing_minutes: float,
    specs: np.ndarray,
    fper: np.ndarray,
    metadata: SpectrumMetadata,
    base: Dict
) -> Tuple:
    """
    Group observations into groups, average each group's spectra, and fit a
    baseline to each group's averaged spectrum. 

    Parameters
    ==========
    dts, spacing_minutes, specs, fper, metadata: see `group_spectra`
    base: see `group_average_baseline_fitting`

    Returns
    =======
    grand_group: list of list of tuple
        See `group_spectra`
    group_avg_spectra: list of tuple
        See `group_averaging`
    group_sg_fits: list of np.ndarray or None
        See `group_average_baseline_fitting`
    """
    grand_group = group_spectra(dts, spacing_minutes, specs, fper, metadata)
    group_avg_spectra = group_averaging(grand_group)
    group_sg_fits = group_average_baseline_fitting(group_avg_spectra, base)
    return grand_group, group_avg_spectra, group_sg_fits
