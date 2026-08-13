"""
Noise
=====

Generates different types of noise and baselines depending on which simualtion is being used.
"""

import numpy as np

def thermal_noise(
    freq: np.ndarray,
    temp: float,
    resistance: float,
    bandwidth: float,
) -> np.ndarray:
    """
    Thermal noise simulation: calculates the johnson-nyquist noise of a component and generates a
    gaussian distribution of noise randomly across frequency space. Only valid in Rayleigh-Jeans
    limit, when h*freq << k*temp.

    Parameters
    ==========
    freq: 1D array
        Frequency axis used in simualting a spectra
    temp: float
        Temperature of component
    resistance: float
        Resistance of the circuit (typically 50 Ohms)
    bandwidth: float
        Bandwidth of the component
    
    Returns
    =======
    johnson_nosie: 1D array
        Johnson-Nyquist noise generated from the gaussian distribution

    Notes
    =====
    Uses the formula: V_noise_rms = sqrt(4*k_b*T*R*B). Bandwidth in this sense is that of the 
    component.
    """
    boltzmann_const = 1.38e-23

    # RMS noise voltage
    voltage_noise_rms = np.sqrt(4 * boltzmann_const * temp * resistance * bandwidth)
    # Generate thermal noise from gaussian distribution
    johnson_noise = np.random.normal(0, voltage_noise_rms, len(freq))

    return johnson_noise

def simulate_baseline(
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Thermal Noise Basline. Black Body noise isn't considered as it is negligably small.

    Parameters
    ==========
    freqs: 1D array
        Frequency axis used in simualting a spectra

    Returns
    =======
    total_power
        Power of thermal baseline

    Notes
    =====
    Uses hardcoded values for resistance, lna bandwidth and amplification, all of which is
    subject to change. Future work should make these user inputable, along wih implimenting
    accurate bandwidths for each component
    """
    resistance = 50 #Ohms
    lna_bandwidth = 19e9
    cavity_power = thermal_noise(freqs, 50e-3, resistance, lna_bandwidth)
    circ_power   = thermal_noise(freqs, 50e-3, resistance, lna_bandwidth)
    a0_1_power   = thermal_noise(freqs, 50e-3, resistance, lna_bandwidth)
    a0_2_power   = thermal_noise(freqs, 300e-3, resistance, lna_bandwidth)
    a0_3_power   = thermal_noise(freqs, 600e-3, resistance, lna_bandwidth)
    lna_power    = thermal_noise(freqs, 7.5, resistance, lna_bandwidth)

    total_power = cavity_power + circ_power + a0_1_power + a0_2_power + a0_3_power + lna_power

    total_amplification = 28 + 43 + 30 - 3 #db
    variable_name = 10**(total_amplification/20)

    total_power *= variable_name

    return total_power

def external_noise(
    freqs_hz: np.ndarray,
    f_start_hz: float,
    f_range: float,
    key: int
) -> np.ndarray:
    """
    Creates a baseline with one of the following functions. Easy to add more functions to test.

    Parameters
    ==========
    freqs_hz: 1D array
        Frequency array of the spectrum the baseline is for
    f_start_hz: float
        Injected axion frequency
    f_range: float
        Range of `freqs_hz`
    key: int
        Decides which function to use

    Returns
    =======
    y: 1D array
        Shaped Basline in accordance to the function picked with `key`
    """
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
    return y
