"""
Downmixing
==========

Mathematically converts a signal from a high frequency to a lower frequency. Applies high and
low pass filters. Doesn't consider phase like a real IQ Mixer (In Phase and Quadrature bands)
"""
from typing import Tuple
import numpy as np

def downmix_signal(
    x_signal: np.ndarray,
    t: np.ndarray,
    f_lo: float,
    hpf: np.ndarray,
    lpf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mathematically converts a signal from a high frequency to a lower frequency. Applies high and
    low pass filters. Doesn't consider phase like a real IQ Mixer (In Phase and Quadrature bands)

    Parameters
    ==========
    x_signal: 1D array
        Wave packet signal generated from the simulation
    t: 1D array
        X axis in time domain
    f_lo: float
        Frequency of local osciallor. Defines the downmixed frequency from f_axion-f_lo
    hpf: 1D array
        High Pass Filter full array
    lpf: 1D array
        LowPass Filter full array

    Returns
    =======
    x_filtered_inverse: 1D array
        Filtered Signal post downmixing
    x_mixed: 1D array
        The mixed signal which is the pure output of a mixer component, which is later filtered
        to get just the signal desired
    """
    x_lo = np.sin(2*np.pi*(f_lo)*t)
    x_mixed = x_signal * x_lo
    # Manual Mixing (Diagnositcs)
    # x_mixed = np.sin(2*np.pi*(freq-freq_local_oscillator)*t)
    #             + np.sin(2*np.pi*(freq+freq_local_oscillator)*t)

    # Apply filter onto linear FFT frequency axis
    x_filtered = np.fft.rfft(x_mixed) * hpf * lpf
    x_filtered = np.fft.irfft(x_filtered)

    return x_filtered, x_mixed
