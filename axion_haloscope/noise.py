import numpy as np

def black_body_radiation(
    freq: np.ndarray,
    temp: float,
) -> np.ndarray:
    """Rayleigh-Jeans approximation, valid when h*freq << k*temp
    """
    c = 3e8
    h = 6.626e-34
    k = 1.38e-23
    R = 50 # Resistance in Ohms
    bandwidth = 19e9


    # RMS noise voltage
    Vn_rms = np.sqrt(4 * k * temp * R * bandwidth)
    # Generate thermal noise samples
    johnson_noise = np.random.normal(0, Vn_rms, len(freq))
            
    return johnson_noise

def simulate_baseline(
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Black Body Noise Basline
    """

    cavity_power = black_body_radiation(freqs, 50e-3)
    circ_power = black_body_radiation(freqs, 50e-3)
    a0_1_power = black_body_radiation(freqs, 50e-3)
    a0_2_power = black_body_radiation(freqs, 300e-3)
    a0_3_power = black_body_radiation(freqs, 600e-3)
    lna_power = black_body_radiation(freqs, 7.5)

    total_power = cavity_power + circ_power + a0_1_power + a0_2_power + a0_3_power + lna_power
    
    total_amplification = 28 + 43 + 30 - 3 #db
    variable_name = 10**(total_amplification/20)

    total_power *= variable_name

    return total_power

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