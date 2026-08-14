"""
Vary Set Size for demonsrative purposes

Tests a range of different set sizes and exports information generated which 
can be passed on to a ploting class 
"""
__all__ = ['evaluate_set_spacing']

from typing import Dict, List, Tuple

import numpy as np

from axion_haloscope.sigma_clipping import claude_clipping, blue_clipping
from axion_haloscope.baseline import remove_baseline


def _init_masks(
    clipping_mode: str,
    set_avg_spectra:np.ndarray,
    sets:List[List[Tuple[np.ndarray, np.ndarray, float]]],
) -> List[List[np.ndarray]]:
    '''Creates empty mask structure and checks the clipping mode here is correct'''
    if clipping_mode == "claude":
        return [np.zeros(len(avg[0]), dtype=int) if avg is not None else None
                for avg in set_avg_spectra]
    elif clipping_mode == "blue":
        return [[np.zeros(len(item[0]), dtype=int) for item in s] for s in sets]
    raise ValueError(f"Unknown clipping_mode: {clipping_mode}")


def _run_clipping(
    clipping_mode: str,
    set_avg_spectra: np.ndarray,
    sets: List[List[Tuple[np.ndarray, np.ndarray, float]]],
    masks: List[List[np.ndarray]],
    fits: List[np.ndarray],
    sigma_cut: int,
    sg_window: int,
    sg_poly: int,
    n_iterations: int,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    '''Runs clipping algorithms and returns masks and fits'''
    for iteration in range(1, n_iterations + 1):
        if clipping_mode == "claude":
            masks, fits = claude_clipping(set_avg_spectra, masks, fits,
                                           sigma_cut, sg_window, sg_poly, iteration)
        else:
            masks, fits = blue_clipping(sets, masks, fits,
                                         sigma_cut, sg_window, sg_poly, iteration)
    return masks, fits


def _masked_fraction(
    clipping_mode: str,
    masks: List[List[np.ndarray]],
) -> Tuple[int, int]:
    '''Calculates and returns the number of masked bins and total bins'''
    if clipping_mode == "claude":
        valid = [m for m in masks if m is not None]
        total_masked = sum(int(np.count_nonzero(m)) for m in valid)
        total_bins   = sum(len(m) for m in valid)
    else:
        total_masked = sum(int(np.count_nonzero(m)) for gm in masks for m in gm)
        total_bins   = sum(len(m) for gm in masks for m in gm)
    return total_masked, total_bins


def _residual_stats(
    clipping_mode: str,
    sets: List[List[Tuple[np.ndarray, np.ndarray, float]]],
    set_avg_spectra: np.ndarray,
    masks: List[List[np.ndarry]],
    fits: List[np.ndarray],
) -> Tuple[List[np.array], List[np.array]]:
    """Return statistics on residuals based on the clipping mode selected"""
    stds, avgs = [], []
    if clipping_mode == "claude":
        for avg, mask, fit in zip(set_avg_spectra, masks, fits):
            if avg is None or mask is None or fit is None:
                continue
            _, spec_avg = avg
            unmasked = mask == 0
            if not unmasked.any():
                continue
            residuals = spec_avg[unmasked] - fit[unmasked]
            stds.append(np.nanstd(residuals))
            avgs.append(np.nanmean(residuals))
    else:
        for set_, set_masks, fit in zip(sets, masks, fits):
            if fit is None or len(set_) == 0:
                continue
            for (spectra, _freq, _res_freq), mask in zip(set_, set_masks):
                unmasked = mask == 0
                if not unmasked.any():
                    continue
                residuals = spectra[unmasked] - fit[unmasked]
                stds.append(np.nanstd(residuals))
                avgs.append(np.nanmean(residuals))
    return stds, avgs


def evaluate_set_spacing(
    spacing: int,
    sets: List[List[Tuple[np.ndarray, np.ndarray, float]]],
    base: Dict[int, int, int, int, float, float, str, int],
    sigma_cut: int,
    n_iterations: int,
) -> Dict[int, int, float, float, float, int, int, float]:
    """
    Take a given set spacing, process the sets as if they are in the actual 
    pipeline, return summary stats for that spacing.

    Parameters
    ----------
    spacing: int, units: Minutes
        maximum time between the start and end of a set - the length of 
        time for which spectra are grouped into sets
    sets: List[...]
        spectra grouped into sets
    base: Dict{...}
        dictionary of all baseline settings from YAML
    sigma_cut: float
        sigma threshold coefficient 
    n_iterations: int
        number of iterations in a clipping loop

    
    Returns
    -------
    Dict{
    spacing_minutes:  int, units: Minutes
        maximum time between the start and end of a set - the length of 
        time for which spectra are grouped into sets
    n_sets: int
        total number of sets 
    average_set_size: float
        average number of spectra in a set
    average_residual_std: float
        average standard deviation of the residuals of all sets
    average_residual_average: float
        average mean of the residuals of all sets
    total_masked: int
        total number of masked bins
    total_bins: int
        total number of bins
    fraction_masked: float
        total_masked / total_bins
    }

    
    Warns
    ------
    If SG fails, warn user that no SG has been fit for that set

    
    See Also
    --------
    axion_haloscope.utils.load_yaml_config for full "base" description
    axion_haloscope.baseline.remove_baseline
    axion_haloscope.sets

    """

    set_avg_spectra = [(np.mean([x[1] for x in s], axis=0),
                        np.mean([x[0] for x in s], axis=0)) for s in sets]
    clipping_mode = base["clipping_mode"].lower()
    set_fits = []
    for _, spec_avg in set_avg_spectra:
        if not spec_avg.any():
            set_fits.append(None)
            continue
        try:
            _, baseline = remove_baseline(spectrum=spec_avg,
                                           window_length=base["sg_window_warm"],
                                           polyorder=base["sg_poly_warm"])
            set_fits.append(baseline)
        except ValueError as e:
            print(f"[Set size variation] spacing={spacing}min: SG fit failed ({e}), skipping set")
            set_fits.append(None)

    masks = _init_masks(clipping_mode, set_avg_spectra, sets)
    masks, set_fits = _run_clipping(clipping_mode, set_avg_spectra, sets, masks, set_fits,
                                    sigma_cut, base["sg_window_warm"], base["sg_poly_warm"],
                                    n_iterations)

    total_masked, total_bins = _masked_fraction(clipping_mode, masks)
    stds, avgs = _residual_stats(clipping_mode, sets, set_avg_spectra, masks, set_fits)

    return {
        "spacing_minutes": spacing,
        "n_sets": len(sets),
        "average_set_size": np.mean([len(s) for s in sets]),
        "average_residual_std": np.mean(stds),
        "average_residual_average": np.mean(np.abs(avgs)),
        "total_masked": total_masked,
        "total_bins": total_bins,
        "fraction_masked": total_masked / total_bins,
    }
