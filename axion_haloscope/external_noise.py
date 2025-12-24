import numpy as np

def external_noise(
    freqs_hz: np.ndarray, f_start_hz: float, f_range: float
) -> np.ndarray:
    x = (freqs_hz - f_start_hz)

    

    y = -2.0*x**3 - 3*x**2 + 12*x + 5
    #y = 8*x**7 - 7*x**6 + 13*x**5 + 16*x**4 - 20*x**3 + 5*x**2 + 11*x + 6
    #y = np.sin(x) + 3
    #y = 1
    return y