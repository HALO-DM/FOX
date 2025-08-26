# tests/test_combine.py
import numpy as np
from axion_haloscope.combine import combine_ml

def test_ml_weights_equal_noise_is_mean():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])
    idx_a = np.arange(0,4); idx_b = np.arange(2,6)  # overlap on 2..3
    combined, sigma, counts = combine_ml(
        [a,b], [idx_a, idx_b], total_rf_bins=6, per_spec_sigma=[1.0,1.0]
    )
    # where both contribute, expect average
    assert np.allclose(combined[2:4], (a[2:4] + b[0:2]) / 2.0)
    assert (counts[2:4] == 2).all()
