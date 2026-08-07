import numpy as np
import sys
from axion_haloscope.baseline import remove_baseline
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List
import warnings

def _sg_masked(freqs, psd, mask_bad, window, order):
    """
    Refit SG using only unmasked bins, then interpolate back onto full grid.
    """
    good = mask_bad == 0
    if good.sum() < window:
        _, baseline = remove_baseline(
                    spectrum=psd,
                    window_length=window,
                    polyorder=order,
                )
        return baseline
    _, fit_good = remove_baseline(
                        spectrum=psd[good],
                        window_length=window,
                        polyorder=order,
                    )
    fit_full = np.interp(freqs, freqs[good], fit_good)
    return fit_full

def _interpolate_nans(y):
    y = np.asarray(y, dtype=float)
    nans = np.isnan(y)
    if nans.any():
        if nans.all():
            return np.nan_to_num(y)  
        x = np.arange(len(y))
        y = y.copy()
        y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    return y

def finalise_specs(mode, group_avg_spectra, groups, group_sg_fits):
    specs, fper = [], []
    if mode == "claude":
        for g, group in enumerate(groups):
            avg = group_avg_spectra[g]
            fit = group_sg_fits[g]
            if avg is None or fit is None:
                continue

            f_grid = avg[0]
            baseline_interp = interp1d(
                f_grid, fit,
                bounds_error=False,
                fill_value=(fit[0], fit[-1]),
            )

            for item in group:
                f_i   = np.asarray(item[1], dtype=float)
                psd_i = np.asarray(item[0], dtype=float)

                bl      = baseline_interp(f_i)
                bl_safe = np.where(np.abs(bl) > 1e-40, bl, np.nanmean(psd_i))
                specs.append(psd_i / bl_safe)
                fper.append(f_i)

    elif mode == "blue":
        for group, baseline in zip(groups, group_sg_fits):
            if baseline is None:
                continue
            group_spectra, group_freqs, _ = map(np.array, zip(*group))
            specs.extend(group_spectra / baseline)
            fper.extend(group_freqs)
    return specs, fper

def claude_clipping(group_avg_spectra, group_masks, group_sg_fits, 
                sigma_cut, sg_window, sg_order, iteration):
    """
    Impliments Claude's Clipping Algorithm. Cleans each set average by performing 
    an SG fit to find a baseline, finding the residuals of that baseline and
    masking any bins that are above/below +-sigma_cut * std. Tracks the 
    iteration this clipping algorithm is happening in, and masks bins 
    accordingly. Mofidied Version of QSHS iterative clipping algorithm:
    https://github.com/QuantumSensorsfortheHiddenSector/DataAnalysis/blob/CM_QSHS_analysis_pipeline/step2_baselineremoval_warmhaystac.py

    Parameters
    ----------
        group_avg_spectra : List[Tuple[ndarray, ndarray]]]
            A grand group that contains all set averages. Each Tuple has 
            X and Y values of 1 set average.
        group_masks : List[ndarray]
            A grand group that contains all set averaged masks.
            Follows same pattern as group_avg_spectra.
        group_sg_fits : List[ndarray]
            A grand group that contains Savitsky Golay fits on set averaged
            spectra.
        sigma_cut : float
            The threshold coefficient
        sg_window : int
            Window length of Savitsky Golay filter
        sg_order : int
            Polynomial Order of Savitsky Golay Filter
        iteration : int
            Iteration number
    Returns
    -------
        new_group_masks : List[List[ndarray]]
            Updated group_masks with new masks from this algorithm
        new_group_sg_fits : List[ndarray]
            Updated group_sg_fits with new fits from this algorithm       
    """
    total_new = 0
    new_group_masks = group_masks.copy()
    new_group_sg_fits = group_sg_fits.copy()
    for g, avg in enumerate(group_avg_spectra):
        if avg is None:
            continue
        f, p         = avg
        current_mask = group_masks[g].copy()
        prev_fit     = group_sg_fits[g].copy()
        if prev_fit is None:
            continue
        new_mask, new_fit, _, _, sigma = general_clipping(p, sg_window, sg_order, sigma_cut, freqs=f, baseline=prev_fit, current_mask=current_mask, iteration=iteration)

        n_new     = int(np.count_nonzero(new_mask == iteration))
        total_new += n_new

        new_group_masks[g]   = new_mask
        new_group_sg_fits[g] = new_fit

    #    print(f"    Group {g:3d}: sigma={sigma:.4g}  "
    #        f"newly masked={n_new:4d}  "
    #        f"total masked={int(np.count_nonzero(new_mask)):4d}/{len(f)}")
    # print(f"  Total newly masked this iteration: {total_new}")
    return new_group_masks, new_group_sg_fits


def blue_clipping(
        groups: List[List[Tuple[np.ndarray, np.ndarray, float]]],
        group_masks: List[List[np.ndarray]],
        group_sg_fits: List[np.ndarray],
        sigma_cut: float,
        sg_window: int, 
        sg_order: int, 
        iteration: int,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """
    Impliments Blue's Clipping Algorithm. Cleans each spectra by performing 
    an SG fit to find a baseline, finding the residuals of that baseline and
    masking any bins that are above/below +-sigma_cut * std. Tracks the 
    iteration this clipping algorithm is happening in, and masks bins 
    accordingly. Takes a new SG fit to pass on.

    Parameters
    ----------
        groups        : List[List[Tuple[ndarray, ndarray, float]]]
            A grand group that contains all sets, each set containing some 
            tuples. Each Tuple has information on 1 spectra.
        group_masks   : List[List[ndarray]]
            A grand group that contains all spectra masks, grouped into sets.
            Follows same pattern as groups
        group_sg_fits : List[ndarray]
            A grand group that contains Savitsky Golay fits on set averaged
            spectra.
        sigma_cut     : float
            The threshold coefficient
        sg_window     : int
            Window length of Savitsky Golay filter
        sg_order      : int
            Polynomial Order of Savitsky Golay Filter
        iteration     : int
            Iteration number
    Returns
    -------
        new_group_masks : List[List[ndarray]]
            Updated group_masks with new masks from this algorithm
        new_group_sg_fits : List[ndarray]
            Updated group_sg_fits with new fits from this algorithm       
    """
    total_new = 0
    new_group_masks = group_masks.copy()
    new_group_sg_fits = group_sg_fits.copy()
    for g, group in enumerate(groups):
        current_masks = group_masks[g].copy()

        n_new = 0
        for spec_idx, (spectra, frequencies, _) in enumerate(group):
            current_mask = current_masks[spec_idx]
            mask, *_ = general_clipping(spectra, sg_window, sg_order, sigma_cut, freqs=frequencies, current_mask=current_mask, iteration=iteration)
            n_new += int(np.count_nonzero(mask == iteration))
            current_masks[spec_idx] = mask

        spectra_stack = np.array([spec for spec, *_ in group])
        mask_stack    = np.array([m != 0 for m in current_masks])

        masked_stack    = np.ma.masked_array(spectra_stack, mask=mask_stack)
        average_spectra = masked_stack.mean(axis=0).filled(np.nan)

        average_for_fit = _interpolate_nans(average_spectra)
        _, new_baseline = remove_baseline(
            average_for_fit, window_length=sg_window, polyorder=sg_order
        )
        new_group_masks[g]   = current_masks
        new_group_sg_fits[g] = new_baseline

        total_new += n_new
        n_bins   = sum(len(m) for m in current_masks)
        n_masked = sum(int(np.count_nonzero(m)) for m in current_masks)
        #print(f"    Group {g+1:3d}: newly masked={n_new:4d}  "
        #      f"total masked={n_masked:4d}/{n_bins}")

    #print(f"  Total newly masked this iteration: {total_new}")
    return new_group_masks, new_group_sg_fits

def general_clipping(
    spectrum: np.ndarray,
    sg_window: int,
    sg_order: int,
    sigma_cut: float,
    freqs: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    current_mask: np.ndarray | None = None,
    iteration: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    A general clipping algorithm. Cleans each spectra by performing 
    an SG fit to find a baseline, finding the residuals of that baseline and 
    masking any bins that are above/below +-sigma_cut * std. Tracks the 
    iteration this clipping algorithm is happening in, and masks bins 
    accordingly.

    Parameters
    ----------
        spectrum      : ndarray, shape (nbins,)
            A single spectrum's Y values
        sg_window    : int
            Window length of Savitsky Golay filter
        sg_order     : int
            Polynomial Order of Savitsky Golay Filter
        sigma_cut    : float
            The threshold coefficient
        freqs        : ndarray | None, shape (n_bins,)
            A single spectrum's X values. If not provided, defaults to
            'np.arange(len(spectrum))'.
        baseline     : ndarray | None
            Pre-Computed specta baseline. If not provided, new SG baseline
            is computed.
        current_mask : ndarray | None
            Pre-Computed mask. If not provided, creates new mask full of 0s 
            with shape (nbins,)
        iteration    : int | None
            Iteration number. If not provided, set to '1'
    Returns
    -------
        mask: ndarray
            Iteration Mapping Mask which keeps track of which iteration an 
            element was masked in (same convention as "current mask")
        baseline: ndarray
            SG baseline that was used to calculate residuals
        residual: ndarray
            Residuals of spectrum - baseline
        threshold: float
            The threshold to mask or not mask a bin
        std: float
            The standard deviation of the residuals

    Warns
    -----
        RuntimeWarning
            Warns if 'current_mask' contains no 0s (i.e every bin is masked).
            If every bin is already masked, `std` would be computed over
            an empty slice (NaN).
    """
    freqs = freqs if (freqs is not None) else np.arange(len(spectrum))
    if current_mask is None:
        current_mask = np.zeros(len(spectrum), dtype=int)
    if (current_mask !=0).all():
        warnings.warn("All bins currently masked", RuntimeWarning)
    if iteration is None:
        iteration = 1
    if baseline is None:
        baseline = _sg_masked(freqs, spectrum, current_mask, sg_window, sg_order)

    residual = spectrum - baseline
    

    std = np.std(residual[current_mask == 0])
    threshold = sigma_cut * std

    new_bad = (current_mask == 0) & (np.abs(residual) > threshold)
    mask = current_mask.copy()
    mask[new_bad] = iteration
    return mask, baseline, residual, threshold, std