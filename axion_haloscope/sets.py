import numpy as np

from axion_haloscope.baseline import remove_baseline

def group_sets(dts, spacing_minutes, specs, fper, metadata):
    sets = []
    n = len(dts)
    threshold = spacing_minutes * 60  # seconds
    i = 0
    while i < n:
        j = i + 1
        while j < n and (dts[j] - dts[i]).total_seconds() < threshold:
            j += 1
        sets.append([(specs[k], fper[k], metadata["res_freq"][k]) for k in range(i, j)])
        i = j
    return sets

# -----------------------------------------------------------------------
# Set Averaging
# -----------------------------------------------------------------------
def set_averaging(sets):
    set_avg_spectra = []
    for s, set in enumerate(sets):
        if set is None:
            set_avg_spectra.append(None)
            continue
        
        set_avg_spectra.append((np.mean([x[1] for x in set], axis=0), np.mean([x[0] for x in set], axis=0)))
    return set_avg_spectra


# -----------------------------------------------------------------------
# Set Average Baseline Fitting
# -----------------------------------------------------------------------
def set_average_baseline_fitting(set_avg_spectra, base):
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

def set_creation(dts, spacing_minutes, specs, fper, metadata, base):
    sets = group_sets(dts, spacing_minutes, specs, fper, metadata)
    set_avg_spectra = set_averaging(sets)
    set_sg_fits = set_average_baseline_fitting(set_avg_spectra, base)
    return sets, set_avg_spectra, set_sg_fits