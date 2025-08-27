# tests/test_lineshape.py
import numpy as np
from axion_haloscope.lineshape import shm_speed_pdf, shm_maxwell_profile, shm_maxwell_template

def test_shm_template_normalizes():
    K =  nine = 9
    T = shm_maxwell_template(K=9, bin_width_hz=1000.0, f0_hz=5.7e9)
    assert np.isclose(T.sum(), 1.0, rtol=1e-6)

def test_profile_support_is_one_sided():
    f0 = 5.7e9
    freqs = f0 + np.linspace(-50000, 50000, 1001)
    P = shm_maxwell_profile(freqs, f0_hz=f0)
    # should be essentially zero below rest-mass frequency
    assert P[freqs < f0].max() <= 1e-12


def test_shm_speed_pdf_normalizes():
    v, p = shm_speed_pdf()
    # non-negative and normalized
    assert np.all(p >= 0)
    assert np.isclose(np.trapz(p, v), 1.0, rtol=1e-6, atol=1e-8)

def test_shm_profile_support_one_sided():
    f0 = 5.7e9
    freqs = f0 + np.linspace(-5e4, 5e4, 2001)  # ±50 kHz around f0
    prof = shm_maxwell_profile(freqs, f0_hz=f0)
    # virtually zero below rest-mass frequency
    assert np.max(prof[freqs < f0]) <= 1e-12
    # non-negative and normalized (discrete)
    assert np.all(prof >= 0)
    s = prof.sum()
    assert np.isclose(s, 1.0, rtol=1e-6, atol=1e-8)


def test_extreme_params_do_not_crash():
    # Very small Earth speed (no boost)
    v, p = shm_speed_pdf(v0=200e3, v_esc=550e3, v_earth=0.0, nv=4096)
    assert np.isfinite(p).all() and np.all(p >= 0)
    assert np.isclose(np.trapz(p, v), 1.0, rtol=1e-5)

    # Large K but reasonable bin width
    T = shm_maxwell_template(K=51, bin_width_hz=1_000.0, f0_hz=5.7e9)
    assert np.isfinite(T).all() and np.all(T >= 0) and np.isclose(T.sum(), 1.0, rtol=1e-6)

def test_profile_monotone_below_f0_zero_above_nonzero():
    f0 = 5.7e9
    freqs = f0 + np.linspace(-20_000, 40_000, 601)
    prof = shm_maxwell_profile(freqs, f0_hz=f0)
    assert np.allclose(prof[freqs < f0], 0.0, atol=1e-12)
    assert prof[freqs >= f0].sum() > 0




'''
this didn't work
def test_template_length_and_sum():
    K =  nine = 9
    T = shm_maxwell_template(K=9, bin_width_hz=1_000.0, f0_hz=5.7e9)
    assert len(T) == 9
    assert np.isclose(T.sum(), 1.0, rtol=1e-6, atol=1e-8)
    assert np.all(T >= 0)

def test_template_peaks_near_center():
    K = 15
    T = shm_maxwell_template(K=K, bin_width_hz=2_000.0, f0_hz=5.7e9)
    c = K // 2
    # power should be concentrated near center bins (not strictly unimodal, but central mass > edges)
    assert T[c] >= T[0] and T[c] >= T[-1]
    assert T[c-1] >= T[1] and T[c+1] >= T[-2]
'''


def test_template_length_and_sum():
    T = shm_maxwell_template(K=9, bin_width_hz=1_000.0, f0_hz=5.7e9)
    assert len(T) == 9
    assert np.all(T >= 0)
    assert np.isclose(T.sum(), 1.0, rtol=1e-6, atol=1e-12)

def test_template_behavior_near_f0():
    # Center index corresponds to f0; SHM profile has zero density exactly at f0.
    K = 15
    T = shm_maxwell_template(K=K, bin_width_hz=2_000.0, f0_hz=5.7e9)
    c = K // 2

    # 1) Exactly at f0 should be ~0
    assert np.isclose(T[c], 0.0, atol=1e-12)

    # 2) Below f0 (left of center) should be ~0 (no support below rest mass)
    assert np.allclose(T[:c], 0.0, atol=1e-12)

    # 3) Just above f0 should be the maximum (first nonzero bin right of center)
    #    and then generally non-increasing as frequency increases.
    right = T[c+1:]
    assert right[0] == np.max(right)  # peak just above f0

    # allow small numerical wiggles; enforce monotonic non-increasing in a relaxed sense
    diffs = np.diff(right)
    # most diffs should be <= tiny positive tolerance
    assert np.count_nonzero(diffs > 1e-10) <= 1  # tolerate one tiny uptick due to interpolation
   
