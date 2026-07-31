import numpy as np
import sys
from axion_haloscope.baseline import remove_baseline
from scipy.interpolate import interp1d
from typing import Optional

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
    if mode == "Claude":
        for g, group in enumerate(groups):
            avg = group_avg_spectra[g]
            fit = group_sg_fits[g]
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

    elif mode == "Blue":
        for group, baseline in zip(groups, group_sg_fits):
            if baseline is None:
                continue
            group_spectra, group_freqs, _ = map(np.array, zip(*group))
            specs.extend(group_spectra / baseline)
            fper.extend(group_freqs)
    return specs, fper

def claude_clipping(group_avg_spectra, group_masks, group_sg_fits, 
                sigma_cut, sg_window, sg_order, iteration):
    total_new = 0
    for g, avg in enumerate(group_avg_spectra):
        if avg is None:
            continue
        f, p         = avg
        current_mask = group_masks[g]
        prev_fit     = group_sg_fits[g]
        if prev_fit is None:
            continue
        new_mask, new_fit, _, _, sigma = general_clipping(p, sg_window, sg_order, sigma_cut, freqs=f, baseline=prev_fit, current_mask=current_mask, iteration=iteration)

        n_new     = int(np.count_nonzero(new_mask == iteration))
        total_new += n_new

        group_masks[g]   = new_mask
        group_sg_fits[g] = new_fit

        print(f"    Group {g:3d}: sigma={sigma:.4g}  "
            f"newly masked={n_new:4d}  "
            f"total masked={int(np.count_nonzero(new_mask)):4d}/{len(f)}")
    print(f"  Total newly masked this iteration: {total_new}")
    return group_masks, group_sg_fits


def blue_clipping(groups, group_masks, group_sg_fits, sigma_cut,
                   sg_window, sg_order, iteration):
    total_new = 0
    for g, group in enumerate(groups):
        current_masks = group_masks[g]

        n_new = 0
        for spec_idx, (spectra, frequencies, res_freq) in enumerate(group):
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
        group_sg_fits[g] = new_baseline

        total_new += n_new
        n_bins   = sum(len(m) for m in current_masks)
        n_masked = sum(int(np.count_nonzero(m)) for m in current_masks)
        print(f"    Group {g+1:3d}: newly masked={n_new:4d}  "
              f"total masked={n_masked:4d}/{n_bins}")

    print(f"  Total newly masked this iteration: {total_new}")
    return group_masks, group_sg_fits

def general_clipping(spectra, sg_window, sg_order, sigma_cut, freqs=None,
                      baseline=None, current_mask=None, iteration=None):
    if current_mask is None:
        current_mask = np.zeros(len(spectra), dtype=int)
    if iteration is None:
        iteration = 1
    if baseline is None:
        baseline = _sg_masked(freqs, spectra, current_mask, sg_window, sg_order)
    freqs = freqs if (freqs is not None) else np.arange(len(spectra))

    residual = spectra - baseline

    std = np.std(residual[current_mask == 0])
    threshold = sigma_cut * std

    new_bad = (current_mask == 0) & (np.abs(residual) > threshold)
    mask = current_mask.copy()
    mask[new_bad] = iteration
    return mask, baseline, residual, threshold, std