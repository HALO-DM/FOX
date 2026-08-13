"""
Filter
======

All representation of physical filters are located here
"""

from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

def pass_filter(
    n_bins: int = 10000000,
    dt: float = 1,
    run_dir: Path="",
    file_name: str="",
    x: np.array = np.array([1,2,3,5,7,10,11,13,14,15,16,17,17.5,18,19,20,21.5,22.5,23,23.5,24,
                            67.5,89.5,111.5,133.5,156.,167,178,189,200]),
    y: np.array = np.array([0.07,0.09,0.12,0.32,0.47,0.65,0.75,1.4,2.93,6.69,11.8,17.23,19.81,
                            22.31,28.01,31.35,37.46,41.58,43.42,45.39,47.26,69.85,68.74,69.81,
                            73.98,74.31,77.32,71.38,75.13,72.96]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a pass filter for a given number of frequency bins, following 2 arrays taken from
    datasheets.

    Parameters
    ==========

    n_bins: int
        Number of bins, used to generate the frequency axis which is passed on to the rest of the
        simulation
    dt: float
        Difference in time between two bins
    run_dir: Path
        Save location of diagnostic graph 
    file_name: str
        Name of figure to save
    x: 1D array
        Array of given frequencies of a filter in MHz (normally found in their datasheet)
        Default is Low Pass Filter
    y: 1D array
        Array of given insertion loss of a filter in dB (normally found in their datasheet)
        Default is Low Pass Filter

    Returns
    =======

    y_linear: 1D array
        Insertion loss converted into linear units of magnitude attenuated, interpolated for the
        number of bins given
    freqs: 1D array
        Frequency axis that has been interpolated to match the number of bins given
        
    See Also
    ========
    https://www.minicircuits.com/pdfs/SLP-10.7+.pdf
        Low pass filter datasheet, used in simulation
    https://media.thorlabs.com/globalassets/items/e/ef/ef5/ef517/ttn051224-s01.pdf?v=0116111839
        High pass filter datasheet, used in simulaion
    """
    freqs = np.fft.rfftfreq(n_bins, d=dt)
    y_linear = 10**(-y/10)  # convert dB to linear power ratio

    # Interpolate in log-x space for smoother results
    x_int = np.logspace(np.log10(x[0]), np.log10(x[-1]), n_bins)

    pchip = PchipInterpolator(x, np.log10(y_linear))
    y_int = 10**pchip(x_int)

    y_linear = 10**pchip(np.clip(freqs/1e6, x[0], x[-1]))

    plt.figure(figsize=(7, 5))
    plt.loglog(x, y_linear, marker='o', linestyle='')
    plt.loglog(x_int, y_int, linestyle='-', linewidth=0.75, color='k')

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude")
    plt.title(file_name)
    plt.tight_layout()
    plt.savefig(run_dir/f"{file_name}", dpi=150)
    plt.close()

    return y_linear, freqs
