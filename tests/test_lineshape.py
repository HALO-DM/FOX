# tests/test_lineshape.py
import numpy as np
from axion_haloscope.lineshape import shm_maxwell_template, shm_maxwell_profile

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
