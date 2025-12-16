import numpy as np

def external_noise(
    freqs_hz: np.ndarray, f_start_hz: float, f_range: float
) -> np.ndarray:
    x = (freqs_hz - f_start_hz)/f_range
    y = 2.0*x**2 + 3*x + 12
    return y