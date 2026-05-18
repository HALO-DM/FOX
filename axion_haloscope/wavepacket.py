import numpy as np

def wavepacket_generation(f0: float, df: float, amplitude: float, n: int, samples_per_cycle:float,) -> np.ndarray:

    dt = 1.0 / (f0 * samples_per_cycle)
    t_vals = np.arange(n) * dt

    indices = np.random.randint(0, n, size=n)
    weight_grid = np.zeros(n, dtype=float)
    np.add.at(weight_grid, indices, np.random.normal(size=n))


    tau = np.arange(n) * dt

    kernel = np.real(np.pi**0.25 * np.sqrt(2.0 * df) * np.exp(-1j * 2 * np.pi * f0 * tau) * np.exp(-0.5 * (df * tau) ** 2))

    # Convolve via FFT
    W = np.fft.rfft(weight_grid)
    H = np.fft.rfft(kernel)
    Xt_vals = np.fft.irfft(W * H, n=n)

    # Normalisation
    V = 0.005383        # Litres
    B = 3               # Tesla
    C = 0.69            # Unitless
    g_ayy = 4e-10       # 1/GeV
    rho_a = 0.45        # GeV/cc
    freq_axion = f0     # Hz
    Q = 2000            # Unitless
    Z = 50              # Ohms

    g_y = g_ayy/5e-14
    power_scaling = 1.79e-21 * (V/200) * (B/7.6)**2 * C * (g_y/0.97)**2 * (rho_a/0.45) * (freq_axion/750e6) * (Q/70000)


    total_amplification = 28 + 43 + 30 - 3 #db
    variable_name = 10**(total_amplification/20)

    amplitude = np.sqrt(power_scaling * Z) * variable_name

        
    rms = np.sqrt(np.mean(Xt_vals**2))
    Xt_vals *= ((1.0 / np.sqrt(2.0)) * amplitude)/ rms 

    return np.vstack((t_vals, Xt_vals)).T