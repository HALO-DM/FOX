import numpy as np

def downmix_signal(x_signal, t, f_lo, hpf, lpf):
    x_lo = np.sin(2*np.pi*(f_lo)*t)
    x_mixed = x_signal * x_lo
    # Manual Mixing
    # x_mixed = np.sin(2*np.pi*(freq-freq_local_oscillator)*t) + np.sin(2*np.pi*(freq+freq_local_oscillator)*t)

    # Apply filter onto linear FFT frequency axis
    X_filtered = np.fft.rfft(x_mixed) * hpf * lpf
    x_filtered = np.fft.irfft(X_filtered)

    return x_filtered, x_mixed