"""
Wavepacket
==========

Generates wavepackets in accordance to the new simulation. 
"""
import numpy as np

def wavepacket_generation(
    f0: float,
    df: float,
    n: int,
    samples_per_cycle:float,
) -> np.ndarray:
    """
    Generates wavepackets at random times with random amplitudes

    Parameters
    ----------
    f0: float
        axion frequency
    df: float
        axion width
    n: int
        number of bins to be produced
    samples_per_cycle: float
        number of bins in a given period
    
    Returns
    -------
    2D array
        frequencies and power distribution of the different spectra

    Notes
    -----

    Values for power equation are currently hardcoded with the values recieved in April 2026.
    Will need updating/chnage to dynamical update in the future.

    """

    dt = 1.0 / (f0 * samples_per_cycle)
    t_vals = np.arange(n) * dt

    indices = np.random.randint(0, n, size=n)
    weight_grid = np.zeros(n, dtype=float)
    np.add.at(weight_grid, indices, np.random.normal(size=n))


    tau = np.arange(n) * dt

    kernel = np.real(np.pi**0.25 * np.sqrt(2.0 * df) * np.exp(-1j * 2 * np.pi * f0 * tau)
                     * np.exp(-0.5 * (df * tau) ** 2))

    # Convolve via FFT
    w = np.fft.rfft(weight_grid)
    h = np.fft.rfft(kernel)
    xt_vals = np.fft.irfft(w * h, n=n)

    # Normalisation
    volume = 0.005383        # Litres
    b_val = 3               # Tesla
    c = 0.69            # Unitless
    g_ayy = 4e-10       # 1/GeV
    rho_a = 0.45        # GeV/cc
    freq_axion = f0     # Hz
    q_factor = 2000            # Unitless
    impedance = 50              # Ohms

    g_y = g_ayy/5e-14
    power_scaling = (1.79e-21 * (volume/200) * (b_val/7.6)**2 * c *
                    (g_y/0.97)**2 * (rho_a/0.45) * (freq_axion/750e6) * (q_factor/70000))


    total_amplification = 28 + 43 + 30 - 3 #db
    variable_name = 10**(total_amplification/20)

    amplitude = np.sqrt(power_scaling * impedance) * variable_name

    rms = np.sqrt(np.mean(xt_vals**2))
    xt_vals *= ((1.0 / np.sqrt(2.0)) * amplitude)/ rms

    return np.vstack((t_vals, xt_vals)).T
