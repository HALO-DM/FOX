import numpy as np
import matplotlib.pyplot as plt

def simulation_stages(freq_axion, freq_local_oscillator,fs, freq_downmixed, n_bins, x_signal, 
                      x_mixed, x_filtered, freqs, psd_filt, mask_show, 
                      H_linear, L_linear, run_dir, t):
    
    X_orig  = np.fft.rfft(x_signal,  n=n_bins)
    X_mixed = np.fft.rfft(x_mixed, n=n_bins)

    psd_orig = (np.abs(X_orig)**2)  / (n_bins * fs)
    psd_mixed = (np.abs(X_mixed)**2) / (n_bins * fs)


    if 2 * (freq_axion + freq_local_oscillator) > fs: #check for aliasing
        combined_freq = fs - (freq_axion + freq_local_oscillator)
        tag = "(Aliased)"
    else:
        combined_freq = (freq_axion + freq_local_oscillator)
        tag = ""


    # Plots 
    fig, axes = plt.subplots(2, 3, figsize=(21, 8))
    fig.suptitle(f"Isolating {freq_downmixed/1e6} MHz Component with {n_bins} bins", fontsize=14, fontweight='bold')

    n_show = -1
    
    axes[0, 0].plot(t[:n_show]*1e6, x_signal[:n_show], color='steelblue', linewidth=0.8)
    axes[0, 0].set(xlabel="Time (μs)", ylabel="Amplitude (V)", title="Time Domain - Original 30GHz Signal")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t[:n_show]*1e6, x_mixed[:n_show], color='purple', linewidth=0.8)
    axes[0, 1].set(xlabel="Time (μs)", ylabel="Amplitude (V)", title="Time Domain - Mixed Signal")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(t[:n_show]*1e6, x_filtered[:n_show], color='darkorange', linewidth=1.2)
    axes[0, 2].set(xlabel="Time (μs)", ylabel="Amplitude (V)", title=f"Time Domain - After Filtering")
    axes[0, 2].grid(True, alpha=0.3)


    axes[1, 0].semilogy(freqs/1e6, psd_orig + 1e-30, color='steelblue')
    axes[1, 0].axvline(x=freq_axion/1e6, color='red', linestyle='dotted', alpha=0.7, label=f'{freq_axion/1e6} MHz')
    axes[1, 0].set(xlabel="Frequency (MHz)", ylabel="PSD", title="FFT - Original (0-200 MHz)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)


    axes[1, 1].semilogy(freqs/1e6, psd_mixed + 1e-30, color='purple')
    axes[1, 1].axvline(x=freq_downmixed/1e6, color='red', linestyle='dotted', alpha=0.7, label=f'{freq_downmixed/1e6} MHz')
    axes[1, 1].axvline(x=(combined_freq)/1e6, color='blue', linestyle='dotted', alpha=0.7, label=f'{(combined_freq)/1e9} GHz {tag}')
    axes[1, 1].set(xlabel="Frequency (MHz)", ylabel="PSD", title="FFT - Mixed Signal")
    axes[1, 1].yaxis.label.set_color("purple")
    axes[1, 1].tick_params(axis='y', colors="purple")
    axes_2 = axes[1, 1].twinx()
    axes_2.set(ylabel="Magnitude")
    axes_2.yaxis.label.set_color("steelblue")
    axes_2.tick_params(axis='y', colors="steelblue")
    lines, labels = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = axes_2.get_legend_handles_labels()
    axes[1, 1].legend(lines + lines2, labels + labels2, loc=0)
    axes[1, 1].grid(True, alpha=0.3)


    axes[1, 2].loglog(freqs[mask_show]/1e6, psd_filt[mask_show] + 1e-30, color='darkorange')
    axes[1, 2].axvline(x=freq_downmixed/1e6, color='red', linestyle='dotted', alpha=0.7, label=f'{freq_downmixed/1e6} MHz')
    axes[1, 2].set(xlabel="Frequency (MHz)", ylabel="PSD", title=f"FFT - After Filtering")
    axes[1, 2].yaxis.label.set_color("darkorange")
    axes[1, 2].tick_params(axis='y', colors="darkorange")
    axes_3 = axes[1, 2].twinx()

    #mask_filter_1 = (freqs >= 0.2e6) & (freqs <= 1e6)  # adjust 5e6 to wherever you want them to stop
    #mask_filter_2 = (freqs >= 1e6) & (freqs <= 200e6)
    axes_3.loglog(freqs/1e6, H_linear, color='steelblue', label="Filter")
    axes_3.loglog(freqs/1e6, L_linear, color='steelblue')

    axes_3.set(ylabel="Magnitude")
    axes_3.yaxis.label.set_color("steelblue")
    axes_3.tick_params(axis='y', colors="steelblue")
    lines, labels = axes[1, 2].get_legend_handles_labels()
    lines2, labels2 = axes_3.get_legend_handles_labels()
    axes[1, 2].legend(lines + lines2, labels + labels2, loc=0)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir/"simulation_stages.png", dpi=150); plt.close()

    return tag, psd_mixed

def aliasing(freqs, psd_mixed, freq_downmixed, fs, combined_freq, tag, run_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(freqs/1e6, psd_mixed + 1e-30, color='purple')
    ax.axvline(x=(freq_downmixed)/1e6, color='red', linestyle='dotted', alpha=0.7, label=f'{freq_downmixed/1e6} MHz')
    ax.axvline(x=(combined_freq)/1e6, color='blue', linestyle='dotted', alpha=0.7, label=f'{(combined_freq)/1e9} GHz {tag}')
    
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"FFT'd Mixed Signal - Sampling Frequency = {fs/1e9} GHz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(run_dir/"aliasing.png", dpi=300); plt.close()