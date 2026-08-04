import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize 
import matplotlib.cm as cm
import pandas as pd


def make_colouriser(values, cmap=plt.cm.viridis, vmin=None, vmax=None):
    finite = values[np.isfinite(values)]
    norm = Normalize(vmin=vmin or np.nanmin(finite), vmax=vmax or np.nanmax(finite))
    def colourise(i):
        v = values[i]
        return "grey" if not np.isfinite(v) else cmap(norm(v))
    return colourise, norm

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


def plot_spectrum(x: np.array,
                y: np.array,
                title: str,
                output_loc: str
                ):
    """
    Plot of a spectrum against frequency
        
    Parameters
    ----------
        x : np.array
            x values to plot.
        y : np.array
            y values to plot.
        title : float
            title of the plot.
        output_loc : string
            location of where the plot will be saved.

    """
    plt.figure(figsize=(13, 7))
    plt.plot(x, y, lw=0.6)
    plt.xlabel("Frequency[GHz]"); plt.ylabel( "Raw Power [arb]")
    plt.title(title); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(output_loc, dpi=150); plt.close()


def vs_time_hist(data: np.array,
                bin_num: int,
                range,
                data_label: str,
                y_label: str,
                title: str,
                output_loc: str):
    """
    Plot of data counts per day.
    
    Parameters
    ----------
        data : 
            values that are being binned.
        bin_num:
            the number of time intervals between the start and end time.
        range :
            the start and end time.
        data_label : string
            label for the data.
        y_label : string
            y axis label.
        title : string
            title for the plot.
        output_loc : string
            location of where the plot will be saved.      
    """
    plt.figure(figsize=(13,7))
    plt.hist(data, bin_num, range, stacked=True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.legend(data_label)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_loc, dpi=150)
    plt.close()


def plot_hist(data: np.array,
            vline: list,
            n: int, 
            bins: int,
            xlabel: str,
            vlabel: list,
            title: str,
            cb_label: str,
            output_loc: str):
    """
    Plot a histogram of the inputted data, optional colour bar and optional vertical lines.

    Parameters
    ----------
        data : np.array
            values that are being binned.
        vline : list
            optional - x value of vertical line to be plotted.
        n : int
            number of different colours required.
        bins : int
            number of bins for the histogram.
        xlabel : float
            label for the x axis
        vlabel: float
            optional - label for the vertical line
        title : string
            title of the plot
        cb_label : string
            optional - label for the colour bar
        output_loc : list
            location of where the plot will be saved.
    
    """
    colors = cm.viridis(np.linspace(0, 1, n))
    fig,ax = plt.subplots(figsize=(13, 7))
    ax.hist(data, bins, stacked=True, color=colors)
    norm2 = mcolors.Normalize(vmin=0, vmax=n - 1)
    sm2 = ScalarMappable(cmap=cm.viridis, norm=norm2)
    sm2.set_array([])

    if n > 1:
        fig.colorbar(sm2, ax=ax, label=cb_label)

    if vline is not None:
        line_colours = cm.tab10(np.linspace(0, 1, len(vline)))
        for line, label, lc in zip(vline, vlabel, line_colours):
            plt.axvline(line, color=lc, alpha=0.8, ls="--", label = f'{label} ({line:.4g})')
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_loc, dpi = 150, bbox_inches='tight')
    plt.close()

def plot_bandwidth(bad_dates_sorted, bad_bandwidths_sorted, good_dates, good_bandwidths, good_order, qc_run_dir, qc):
    plt.figure(figsize=(13, 7))
    plt.scatter(bad_dates_sorted, bad_bandwidths_sorted, color="firebrick", label="removed (below min threshold)")
    plt.scatter(good_dates[good_order], good_bandwidths[good_order], color="steelblue", alpha=0.6, label="kept (above min threshold)")
    plt.axhline(qc["bw_min"], color="black", linestyle="--", linewidth=1, label=f"bw_min = {qc['bw_min']:.4g} Hz")
    plt.xlabel("Date")
    plt.ylabel("Bandwidth [Hz]")
    plt.title("Removed spectra: bandwidth below threshold")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.ylim(0, 0.006)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(qc_run_dir / "bad_bandwidth_vs_time.png", dpi=150)
    plt.close()


def plot_events_against_time(invalid_dates, valid_dates, all_dates, count_invalid, count_valid, count_all, raw_run_dir):
        # Plot of total number of events againist time
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(invalid_dates, count_invalid, label='invalid files', color="red")

    ax.plot(valid_dates, count_valid, label="valid files", color="green")
    ax.plot(all_dates, count_all, label='all files', linestyle='dashed', color="orange")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.set_xlabel("Date-Time")
    ax.set_ylabel("Events")
    ax.set_title(f"Evolution of number of events w.r.t. time")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{raw_run_dir}/events_against_time.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_rms_against_time(sset, qc_run_dir):
    # Plot rms evolution with time - data
    rms_vals = []
    dates = sset.metadata["date"]
    dates = pd.to_datetime(dates)

    for s in sset.spectra:
        med = np.nanmedian(s)
        rms = np.sqrt(np.nanmean((s - med) ** 2))
        rms_vals.append(rms)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.scatter(dates, rms_vals, marker=".")
    ax.set_xlabel("Date"); ax.set_ylabel("rms values [arb]")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")
    ax.set_title("rms over time (data)"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(qc_run_dir/"rms_against_time_data.png", dpi=150); plt.close()

def plot_spectra(fper, specs, count, max_plots, run_dir, step, title, file_name, offset=None):
    if offset == None:
        offset = np.zeros(len(fper))
    plt.figure(figsize=(13, 7))
    for i, (freqs, spec) in enumerate(zip(fper, specs)):
        if i % step != 0:
            continue
        if max_plots is not None and count >= max_plots:
            break
        plt.plot(freqs/1e9 + offset[i], spec, lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title(title); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir/file_name, dpi=150); plt.close()

def plot_exclusion(freqs_r_hz, gmin, outfile=None, title="95% CL Exclusion (toy)"):
    plt.figure(figsize=(9,4))
    plt.plot(np.asarray(freqs_r_hz)/1e9, gmin, lw=1.5)
    plt.xlabel("Frequency [GHz]"); plt.ylabel(r"$g_{a\gamma\gamma}$ (arb vs $g_0$)")
    plt.title(title); plt.grid(alpha=0.3)
    if outfile:
        plt.tight_layout(); plt.savefig(outfile, dpi=160)
    return plt.gca()

def plot_scatter(res_freq_diff, raw_run_dir):
    plt.figure(figsize=(13, 7))
    plt.scatter(range(len(res_freq_diff)),res_freq_diff)
    plt.xlabel("Spectrum Index"); plt.ylabel("Resonance Frequency Offset [Hz]")
    plt.title("Resonance Frequency Offset vs Spectrum Index"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(raw_run_dir/"res_freq_offset_vs_index.png", dpi=150); plt.close()

def plot_evo_of_freq(colour_vals, metadata_dates, cbar_label, raw_run_dir):
    fig, ax = plt.subplots(figsize = (13,7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.plot(metadata_dates, colour_vals)
    ax.set_xlabel("Date-Time")
    ax.set_ylabel(cbar_label)
    ax.set_title(f"Evolution of {cbar_label} w.r.t. Time")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{raw_run_dir}/evolution_of_frequency.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_sets(mode, fper, specs, colour_vals, cbar_label, run_dir, title, file_name, cmap, set_sg_fits=None, sets=None):
    '''
    Optimise Later
    '''


    colourise, norm = make_colouriser(colour_vals, cmap=plt.cm.viridis)
    fig, ax = plt.subplots(figsize = (13,7))
    if mode == "baseline_removal":
        for spec, freq, cv in zip(specs, fper, colour_vals):
            ax.plot(freq, spec, linestyle="", marker="o", markersize=3, color=colourise(cv), alpha=0.7)
        ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    elif mode == "sg_fit":
        for s, (freqs, specs, fit) in enumerate(zip(fper, specs, set_sg_fits)):
            ax.plot(freqs/1e6, specs,lw=1.0, alpha=0.55, color=colourise(s), label=f"Set {s}")
            ax.plot(freqs/1e6, fit, lw=1.8, alpha=0.95, color=colourise(s), linestyle="--")
    elif mode == "sets":
        for s, set in enumerate(sets):
            ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.mean([x[0] for x in set], axis=0), alpha=0.8, color=colourise(s), label =f"Set {s}")
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label)

    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/{file_name}", dpi=150, bbox_inches='tight')
    plt.close()
