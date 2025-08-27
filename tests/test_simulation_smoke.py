import numpy as np
from axion_haloscope.simulation import simulate_spectra, AxionParams

def test_simulation_shapes_and_seed():
    specs1, fper1, rf1, map1 = simulate_spectra(
        n_spectra=4, n_bins=2000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80,
        noise_sigma=1.0, rng_seed=1234, axion=None
    )
    specs2, fper2, rf2, map2 = simulate_spectra(
        n_spectra=4, n_bins=2000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80,
        noise_sigma=1.0, rng_seed=1234, axion=None
    )

    assert len(specs1) == 4 and specs1[0].shape == (2000,)
    assert rf1.ndim == 1 and rf1.size == 2000 + 3*80  # global grid size
    # deterministic with same seed
    for a,b in zip(specs1, specs2):
        assert np.allclose(a, b)
    # RF bookkeeping consistent
    for i, idx in enumerate(map1):
        assert np.allclose(fper1[i], rf1[idx])



def test_axion_injection_produces_local_excess():
    n_spectra, n_bins, bw = 6, 4000, 100.0
    f_start, step = 5.70e9, 80
    # Put axion roughly mid-span
    rf_total_bins = n_bins + (n_spectra-1)*step
    f_ax = f_start + 0.5*rf_total_bins*bw

    specs, fper, rf, rf_map = simulate_spectra(
        n_spectra=n_spectra, n_bins=n_bins, bin_width_hz=bw,
        f_start_hz=f_start, tune_step_bins=step, rng_seed=7,
        axion=AxionParams(f_axion_hz=f_ax, sigma_hz=2500.0, total_power=30.0)
    )

    # Naive RF average (no baseline removal), just to "see" the bump
    summed = np.zeros_like(rf, dtype=float)
    counts = np.zeros_like(rf, dtype=int)
    for s, idx in zip(specs, rf_map):
        summed[idx] += s
        counts[idx] += 1
    avg = np.divide(summed, counts, where=counts>0)

    # Build an in-bounds signal window W and a background window Wa far away
    i_ax = int(np.argmin(np.abs(rf - f_ax)))
    # half-width ~ 15 kHz window -> bins:
    half_w = max(1, int(15_000.0 / bw))
    a = max(0, i_ax - half_w)
    b = min(len(rf), i_ax + half_w + 1)
    W = np.zeros_like(rf, dtype=bool); W[a:b] = True

    # background window: shift by ~30*half_w bins (far but in-bounds)
    shift = 30 * half_w
    a_bg = a + shift
    b_bg = b + shift
    if b_bg > len(rf):  # if off the right edge, move to the left instead
        a_bg = max(0, a - shift)
        b_bg = max(b - shift, a_bg + 1)
    Wa = np.zeros_like(rf, dtype=bool); Wa[a_bg:b_bg] = True

    # sanity: both windows non-empty and disjoint
    assert W.any() and Wa.any() and not np.any(W & Wa)

    assert np.mean(avg[W]) > np.mean(avg[Wa])

