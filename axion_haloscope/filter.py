import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def pass_filter(N: int = 10000000,
    dt: float = 1,
    run_dir: str="",
    name: str="",
    y: np.array = np.array([0.07,0.09,0.12,0.32,0.47,0.65,0.75,1.4,2.93,6.69,11.8,17.23,19.81,22.31,28.01,31.35,37.46,41.58,43.42,45.39,47.26,69.85,68.74,69.81,73.98,74.31,77.32,71.38,75.13,72.96]),
    x: np.array = np.array([1,2,3,5,7,10,11,13,14,15,16,17,17.5,18,19,20,21.5,22.5,23,23.5,24,67.5,89.5,111.5,133.5,156.,167,178,189,200]),
    ):
    freqs = np.fft.rfftfreq(N, d=dt)
    y_linear = 10**(-y/10)  # convert dB to linear power ratio

    # Interpolate in log-x space for smoother results
    x_int = np.logspace(np.log10(x[0]), np.log10(x[-1]), N)

    pchip = PchipInterpolator(x, np.log10(y_linear))
    y_int = 10**pchip(x_int)


    plt.figure(figsize=(7, 5))

    H_linear = 10**pchip(np.clip(freqs/1e6, x[0], x[-1]))
    plt.loglog(x, y_linear, marker='o', linestyle='')
    plt.loglog(x_int, y_int, linestyle='-', linewidth=0.75, color='k')

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(run_dir/f"{name}.png", dpi=150); plt.close()

    return H_linear, freqs
