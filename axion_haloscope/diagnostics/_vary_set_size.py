__all__ = ['evaluate_set_spacing']

import numpy as np

from axion_haloscope.sigma_clipping import claude_clipping, blue_clipping
from axion_haloscope.baseline import remove_baseline

def _init_masks(clipping_mode, set_avg_spectra, var_sets):
    if clipping_mode == "Claude":
        return [np.zeros(len(avg[0]), dtype=int) if avg is not None else None
                for avg in set_avg_spectra]
    elif clipping_mode == "Blue":
        return [[np.zeros(len(item[0]), dtype=int) for item in s] for s in var_sets]
    raise ValueError(f"Unknown clipping_mode: {clipping_mode}")


def _run_clipping(clipping_mode, set_avg_spectra, var_sets, masks, fits,
                   sigma_cut, sg_window, sg_poly, n_iterations):
    for iteration in range(1, n_iterations + 1):
        if clipping_mode == "Claude":
            masks, fits = claude_clipping(set_avg_spectra, masks, fits,
                                           sigma_cut, sg_window, sg_poly, iteration)
        else:
            masks, fits = blue_clipping(var_sets, masks, fits,
                                         sigma_cut, sg_window, sg_poly, iteration)
    return masks, fits


def _masked_fraction(clipping_mode, masks):
    if clipping_mode == "Claude":
        valid = [m for m in masks if m is not None]
        total_masked = sum(int(np.count_nonzero(m)) for m in valid)
        total_bins   = sum(len(m) for m in valid)
    else:
        total_masked = sum(int(np.count_nonzero(m)) for gm in masks for m in gm)
        total_bins   = sum(len(m) for gm in masks for m in gm)
    return total_masked, total_bins


def _residual_stats(clipping_mode, var_sets, set_avg_spectra, masks, fits):
    stds, avgs = [], []
    if clipping_mode == "Claude":
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
        for set_, set_masks, fit in zip(var_sets, masks, fits):
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


def evaluate_set_spacing(spacing, var_sets, base, sigma_cut, n_iterations):
    """Fit + clip a single spacing's sets, return summary stats for that spacing."""
    set_avg_spectra = [(np.mean([x[1] for x in s], axis=0),
                        np.mean([x[0] for x in s], axis=0)) for s in var_sets]

    set_fits = []
    for freqs_avg, spec_avg in set_avg_spectra:
        if not spec_avg.any():
            set_fits.append(None)
            continue
        try:
            _, baseline = remove_baseline(spectrum=spec_avg,
                                           window_length=base["sg_window_warm"],
                                           polyorder=base["sg_poly_warm"])
            set_fits.append(baseline)
        except Exception as e:
            print(f"[Set size variation] spacing={spacing}min: SG fit failed ({e}), skipping set")
            set_fits.append(None)

    masks = _init_masks(base["clipping_mode"], set_avg_spectra, var_sets)
    masks, set_fits = _run_clipping(base["clipping_mode"], set_avg_spectra, var_sets, masks, set_fits,
                                     sigma_cut, base["sg_window_warm"], base["sg_poly_warm"], n_iterations)

    total_masked, total_bins = _masked_fraction(base["clipping_mode"], masks)
    stds, avgs = _residual_stats(base["clipping_mode"], var_sets, set_avg_spectra, masks, set_fits)

    return {
        "spacing_minutes": spacing,
        "n_sets": len(var_sets),
        "average_set_size": np.mean([len(s) for s in var_sets]),
        "average_residual_std": np.mean(stds),
        "average_residual_average": np.mean(np.abs(avgs)),
        "total_masked": total_masked,
        "total_bins": total_bins,
        "fraction_masked": total_masked / total_bins,
    }