import numpy as np

def external_noise(
    freqs_hz: np.ndarray, f_start_hz: float, f_range: float, key: int
) -> np.ndarray:
    x = (freqs_hz - f_start_hz)/f_range

    if key == 1:
        y = x + 4
    elif key == 2:
        y = x**2 + 1
    elif key == 3:
        y = -2.0*x**3 - 3*x**2 + 12*x + 5
    elif key == 4:
        y = 8*x**7 - 7*x**6 + 13*x**5 + 16*x**4 - 20*x**3 + 5*x**2 + 11*x + 6
    elif key == 5:
        y = np.exp(0.5 * x) + 1
    elif key == 6:
        y = 1.5/(x**2 + 0.5) + 1
    elif key == 7:
        y = 1.2* 0.8 * np.exp(-0.2 * x **2) * (1 + 0.5 * np.sin(3 * x))
    elif key == 8:
        y = np.log(x**2 + 3) + np.exp(-0.5*x**2) + 1
    elif key == 9:
        y = np.exp(0.1 * x) + 0.5 * np.cos(2*x) + 1
    else:
        y = 1
    #print("key = ", key)
    return y