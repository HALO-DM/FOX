import numpy as np
import sys
from axion_haloscope.baseline import remove_baseline

def _sg_masked(freqs, psd, mask_bad, window, order):
    """
    Refit SG using only unmasked bins, then interpolate back onto full grid.
    """
    good = ~mask_bad
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

def claude_clipping(group_avg_spectra, group_masks, group_sg_fits, 
                    sigma_cut, sg_window, sg_order):
    total_new = 0
    for g, avg in enumerate(group_avg_spectra):
        if avg is None:
            continue
        f, p         = avg
        current_mask = group_masks[g]
        prev_fit     = group_sg_fits[g]
        if prev_fit is None:
            continue

        residuals = p - prev_fit
        sigma     = np.std(residuals[~current_mask])
        new_bad   = np.abs(residuals) > sigma_cut * sigma
        combined  = current_mask | new_bad
        n_new     = int(np.sum(new_bad & ~current_mask))
        total_new += n_new

        new_fit = _sg_masked(f, p, combined, sg_window, sg_order)

        group_masks[g]   = combined
        group_sg_fits[g] = new_fit

        print(f"    Group {g:3d}: sigma={sigma:.4g}  "
              f"newly masked={n_new:4d}  "
              f"total masked={int(combined.sum()):4d}/{len(f)}")
    print(f"  Total newly masked this iteration: {total_new}")
    return group_masks, group_sg_fits


def blue_clipping(groups, masked_total, sigma_cut, persistent_masks, sg_window, sg_order):
    
    masked_by_group = []
    new_groups = []
    new_group_sg_fits = []
    new_persistent_masks = []

    for group_idx, group in enumerate(groups):
        group_masks = persistent_masks[group_idx]   

        spectra_stack = np.array([x[0] for x in group])
        mask_stack    = np.array(group_masks)

        masked_stack = np.ma.masked_array(spectra_stack, mask=mask_stack)
        average_spectra = masked_stack.mean(axis=0).filled(np.nan)
        sd_spectra      = masked_stack.std(axis=0).filled(np.nan)

        average_for_fit = _interpolate_nans(average_spectra)
        #baseline = _sg_masked(group[1][0], average_for_fit, masked_stack, sg_window, sg_order)
        _, baseline = remove_baseline(
                    average_for_fit,
                    window_length=sg_window,
                    polyorder=sg_order,
                )
        new_group_sg_fits.append(baseline)

        masked_new = []
        new_group = []
        new_group_masks = []

        for spec_idx, (spectra, frequencies, res_freq) in enumerate(group):
            prev_mask = group_masks[spec_idx]

            deviation = np.abs(spectra - baseline)
            new_flags = (deviation > sigma_cut * sd_spectra) & ~prev_mask

            cum_mask = prev_mask | new_flags

            spec_m = np.ma.masked_array(spectra, cum_mask)
            freq_m = np.ma.masked_array(frequencies, cum_mask)

            cleaned_spec = _interpolate_nans(spec_m.filled(np.nan))
            cleaned_freq = _interpolate_nans(freq_m.filled(np.nan))

            newly_masked_idx = np.where(new_flags)[0]
            for idx in newly_masked_idx:
                masked_new.append([frequencies[idx], spectra[idx]])

            new_group.append([cleaned_spec, cleaned_freq, res_freq])
            new_group_masks.append(cum_mask)

        masked_by_group.append(masked_new)
        new_groups.append(new_group)
        new_persistent_masks.append(new_group_masks)

    for g in range(len(groups)):
        masked_total[g].extend(masked_by_group[g])
        groups = new_groups
        group_sg_fits = new_group_sg_fits
        persistent_masks = new_persistent_masks
    return persistent_masks, group_sg_fits, groups, masked_total, masked_by_group