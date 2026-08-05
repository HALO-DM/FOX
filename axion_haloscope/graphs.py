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


    colourise, norm = make_colouriser(colour_vals, cmap=cmap)
    fig, ax = plt.subplots(figsize = (13,7))
    if mode == "baseline_removal":
        for s, (spec, freq) in enumerate(zip(specs, fper)):
            ax.plot(freq, spec, linestyle="", marker="o", markersize=3, color=colourise(s), alpha=0.7)
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


def plot_iteritive_clipping(set_avg_spectra, plotting_set_masks, set_sg_fits,iteration, run_dir, set_mean_res):
    fig, ax = plt.subplots(figsize=(13, 7))
    colourise_1, norm1 = make_colouriser(set_mean_res, cmap=plt.cm.viridis, vmin=None, vmax=None)
    colourise_2, norm2 = make_colouriser(set_mean_res, cmap=plt.cm.inferno, vmin=None, vmax=None)
    for s, avg in enumerate(set_avg_spectra):
        if avg is None:
            continue
        freqs, specs  = avg
        masks  = plotting_set_masks[s]
        fit   = set_sg_fits[s]

        for mask in masks:

            unmasked = mask == 0
            masked_this_iteration = mask == iteration
            masked_previously = (mask > 0) & (mask != iteration)
            if masked_this_iteration.any():
                ax.scatter(freqs[masked_this_iteration]/1e6, specs[masked_this_iteration], marker = ".", color=colourise_2(s), zorder=5)

            if masked_previously.any():
                ax.scatter(freqs[masked_previously]/1e6, specs[masked_previously], c="grey", zorder=4)

        ax.plot(freqs[unmasked]/1e6, specs[unmasked],lw=1.0, alpha=0.55, color=colourise_1(s), label=f"Set {s}")
        ax.plot(freqs/1e6, fit, lw=1.8, alpha=0.95, color=colourise_1(s), linestyle="--")

    sm_res1 = ScalarMappable(cmap=plt.cm.viridis, norm=norm1)
    sm_res1.set_array([])
    fig.colorbar(sm_res1, ax=ax, label="Mean cavity resonance  [GHz]", pad=0.02)

    sm_res2 = ScalarMappable(cmap=plt.cm.inferno, norm=norm2)
    sm_res2.set_array([])
    fig.colorbar(sm_res2, ax=ax, label="Mean cavity resonance  [GHz]", pad=0.10)

    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Set-averaged spectra with initial SG fits  (dashed = fit)")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/masked_bin_iteration_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_3x3(mode, sets, set_mean_res, xlabel, ylabel, title, file_name, run_dir):
    colourise, norm = make_colouriser(set_mean_res, cmap=plt.cm.viridis, vmin=None, vmax=None)
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(26, 10))
    axes_flat = axes.flatten()
    sets_per_subplot = 3
    for ax_idx, ax in enumerate(axes_flat):
        start = ax_idx *sets_per_subplot
        end = start + sets_per_subplot
        for s in range(start, min(end, len(sets))):
            if mode == "set_average":
                ax.plot(
                    np.mean([x[1] for x in sets[s]], axis=0) / 1e6,
                    np.mean([x[0] for x in sets[s]], axis=0),
                    alpha=0.8, color=colourise(s), label=f"Set {s}")
            elif mode == "std":
                ax.plot(
                    np.mean([x[1] for x in sets[s]], axis=0) / 1e6,
                    np.std([x[0] for x in sets[s]], axis=0),
                    alpha=0.8, color=colourise(s), label=f"Set {s}")

    for row in range(3):
        axes[row, 0].set_ylabel(xlabel)
    for col in range(3):
        axes[2, col].set_xlabel(ylabel)

    sm_res = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm_res.set_array([])
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.colorbar(sm_res, ax=axes_flat, label="Mean cavity resonance  [GHz]")

    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes_flat]
    left = min(p.x0 for p in positions)
    right = max(p.x1 for p in positions)
    center_x = (left + right) / 2

    fig.suptitle(title, fontsize=32, x=center_x)
    plt.savefig(f"{run_dir}/{file_name}", dpi=150, bbox_inches='tight')
    plt.close()

def plot_std_freq(sets, set_mean_res, run_dir):
    colourise, norm = make_colouriser(set_mean_res, cmap=plt.cm.viridis, vmin=None, vmax=None)
    fig, ax = plt.subplots(figsize=(13, 7))
    for s, set in enumerate(sets):
        ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.std([x[0] for x in set], axis=0), alpha=0.8, color=colourise(s), label =f"Set {s}")
    sm_res = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("Standard deviation  [V²/Hz]")
    ax.set_title(f"Standard deviation of averaged spectra againist frequency - all sets (n = {len(sets)})")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/std_vs_freq_all.png", dpi = 150, bbox_inches='tight')
    plt.close()

def plot_std_set_num(av_stds, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.scatter(range(0, len(av_stds)), av_stds)
    ax.set_xlabel("Set number")
    ax.set_ylabel("Standard deviation  [V²/Hz]")
    ax.set_title("Average standard deviation per set againist set number")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/std_vs_set_num.png", dpi = 150, bbox_inches='tight')
    plt.close()

def plot_spectra_in_set(set, s, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    greys = cm.Greys(np.linspace(0.3, 0.9, len(set)))
    for i, x in enumerate(set):
        ax.plot(x[1]/1e6, x[0], color=greys[i])
    ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.mean([x[0] for x in set], axis=0), alpha=0.8, color="red", label="set averaged")
    norm = mcolors.Normalize(vmin=0, vmax=len(set))
    sm = ScalarMappable(cmap=cm.Greys, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Spectrum index in set")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Set-averaged spectra and the individual spectra — set {s} (n={len(set)})")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{run_dir}/set_and_average_spectra_{s}.png", dpi = 150, bbox_inches='tight')
    plt.close()

def plot_log_spectra_in_set(set, s, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    greys = cm.Greys(np.linspace(0.3, 0.9, len(set)))
    for i, x in enumerate(set):
        ax.plot(x[1]/1e6, np.log(x[0]), color=greys[i])
    ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.log(np.mean([x[0] for x in set], axis=0)), alpha=0.8, color="red", label="set averaged")
    norm = mcolors.Normalize(vmin=0, vmax=len(set))
    sm = ScalarMappable(cmap=cm.Greys, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Spectrum index in set")
    # ax.set_yscale("log")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"log set-averaged spectra and the log individual spectra — set {s} (n = {len(set)})")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{run_dir}/log_set_and_average_spectra_{s}.png", dpi = 150, bbox_inches='tight')
    plt.close()

def plot_set_average_errors(set, s, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.errorbar(np.mean([x[1] for x in set], axis=0)/1e6, np.mean([x[0] for x in set], axis=0), np.std([x[0] for x in set], axis=0), alpha=0.5, ecolor="blue", color="red", label="std of average")
    ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.mean([x[0] for x in set], axis=0), alpha=0.8, color='red', label="set averaged")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Set-averaged spectra with errors — set {s}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{run_dir}/set_averaged_spectra_errors_{s}.png", dpi = 150, bbox_inches='tight')
    plt.close()

def plot_zoom_set_average_errors(set, s, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.errorbar(np.mean([x[1] for x in set], axis=0) / 1e6, np.mean([x[0] for x in set], axis=0), np.std([x[0] for x in set], axis=0), alpha=0.5, ecolor="blue", color="red", label="std of average")
    ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.mean([x[0] for x in set], axis=0), alpha=0.8, color='red', label="set averaged")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Set-averaged spectra with errors — set {s} (zoomed)")

    x_min, x_max = 1.5, 1.75
    ax.set_xlim(x_min, x_max)
    freqs_avg = np.mean([x[1] for x in set], axis=0) / 1e6
    spec_avg = np.mean([x[0] for x in set], axis=0)
    spec_std = np.std([x[0] for x in set], axis=0)

    in_range = (freqs_avg >= x_min) & (freqs_avg <= x_max)
    if in_range.any():
        y_lower = np.min(spec_avg[in_range] - spec_std[in_range])
        y_upper = np.max(spec_avg[in_range] + spec_std[in_range])
        y_pad = 0.05 * (y_upper - y_lower)
        ax.set_ylim(y_lower - y_pad, y_upper + y_pad)

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{run_dir}/set_averaged_spectra_errors_zoom{s}.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_std_against_freq(set, s, set_mean_res, run_dir):
    colourise, norm = make_colouriser(set_mean_res, cmap=plt.cm.viridis, vmin=None, vmax=None)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(np.mean([x[1] for x in set], axis=0)/1e6, np.std([x[0] for x in set], axis=0), alpha=0.8, color=colourise(s), label =f"Set {s}")
    sm_res = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("Standard deviation  [V²/Hz]")
    ax.set_title(f"Standard deviation of set average againist frequency - set {s}")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/std_vs_freq_{s}.png", dpi = 150, bbox_inches='tight')
    plt.close()

def plot_claude_residuals(freqs, residuals, s, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(freqs/1e6 ,residuals)
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("Residuals  [V²/Hz]")
    ax.set_title(f"Residuals - set {s} (Claude's clipping method)")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/claude_residuals_{s}.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_blue_residuals(set, fit, colours, s, run_dir):
    all_residuals = []
    fig, ax = plt.subplots(figsize=(13, 7))
    for spec_idx, (spectra, frequencies, res_freq) in enumerate(set):
        residuals = spectra - fit
        all_residuals.append(residuals)
        ax.plot(frequencies / 1e6, residuals, lw=0.8, alpha=0.7, color=colours[spec_idx])


    norm = mcolors.Normalize(vmin=0, vmax=n - 1)
    sm = ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Spectrum index in set")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("Residuals  [V²/Hz]")
    ax.set_title(f"Residuals — set {s} (Blue's clipping method)")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/blue_residuals_{s}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return all_residuals

def plot_combination(rf, combined, run_dir):    
    plt.figure(figsize=(13, 7))
    plt.plot(rf/1e9, combined, lw=0.8, color="black", label="combined")
    plt.title("Combined spectrum (baseline-removed)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Excess power [arb]"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run_dir/"combined.png", dpi=150); plt.close()

def plot_grand_spectrum(freqs_r, z, run_dir):
    plt.figure(figsize=(13, 7))
    plt.plot(freqs_r/1e9, z, lw=0.8)
    plt.title("Grand spectrum z-score (SHM matched filter)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("z"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run_dir/"grand_z.png", dpi=150); plt.close()

def plot_candidates(freqs_r, zvals, theta, cands, run_dir):
    fig, ax = plt.subplots(figsize=(13, 7))

    # plot the z-score trace
    ax.plot(freqs_r/1e9, zvals, lw=0.7, label="z-score")

    # detection threshold line
    ax.axhline(theta, color="tab:red", ls="--", label=f"threshold ({theta:.2f}σ)")
    ax.axhline(3, color="tab:orange", ls="--", label=f"Observation (3σ)")
    ax.axhline(5, color="tab:purple", ls="--", label=f"Discovery (5σ)")

    # mark candidate points
    if len(cands) > 0:
        ax.scatter(freqs_r[cands]/1e9, zvals[cands],
                   color="tab:orange", s=30, zorder=5, label="candidates")

    ax.set(xlabel="Frequency [GHz]", ylabel="z",
           title="Grand spectrum with candidate markers")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir/"candidates.png", dpi=150)
    plt.close(fig)

def plot_data_cleaning(freq, spec,metadata, baseline, threshold, residuals, spec_idx, masked_this_iteration, masked_previously, mask, unmasked, base, iteration, run_dir):

    res_freq_array = np.asarray(metadata["res_freq"], dtype=float)

    colouriser, norm =make_colouriser(res_freq_array, cmap=plt.cm.viridis, vmin=None, vmax=None)
    color = colouriser(spec_idx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True,
                                                        gridspec_kw={"height_ratios": [2, 1]})
                    
    if masked_this_iteration.any():
        ax1.scatter(freq[masked_this_iteration]/1e6, spec[masked_this_iteration],
                marker=".", color="red", zorder=5, label=f"{np.count_nonzero(mask == iteration)} Bins Masked this iteration")
        ax2.scatter(freq[masked_this_iteration]/1e6, residuals[masked_this_iteration],
                marker=".", color="red", zorder=5, label=f"{np.count_nonzero(mask == iteration)} Bins Masked this iteration")
        
    if masked_previously.any():
        ax1.scatter(freq[masked_previously]/1e6, spec[masked_previously],
                marker=".", c="grey", zorder=4, label=f"{np.count_nonzero((mask > 0) & (mask != iteration))} Bins Previously Masked ")
        ax2.scatter(freq[masked_previously]/1e6, residuals[masked_previously],
                marker=".", c="grey", zorder=4, label=f"{np.count_nonzero((mask > 0) & (mask != iteration))} Bins Previously Masked ")

    ax1.plot(freq[unmasked]/1e6, spec[unmasked], lw=1.0, alpha=0.75, color=color, label="Post-Mask Data")
    ax1.plot(freq/1e6, spec, lw=1.0, alpha=0.35, color=color, label="Raw Data")
    ax1.plot(freq/1e6, baseline, lw=1.8, alpha=0.95, color=color, linestyle="--", label="Baseline")

    ax2.plot(freq[unmasked]/1e6, residuals[unmasked], lw=1.0, alpha=0.75, color=color, label="Masked Residuals")
    ax2.plot(freq/1e6, residuals, lw=1.0, alpha=0.35, color=color, label="Unmasked Residuals")
    ax2.axhline(threshold, alpha=0.35, color="red", linestyle="dashed",label=f"{base["sigma_cut"]}σ Threshold")
    ax2.axhline(-threshold, alpha=0.35, color="red", linestyle="dashed")
    max_vals = [np.max(spec[masked_this_iteration])] if masked_this_iteration.any() else []
    if masked_previously.any():
        max_vals.append(np.max(np.abs(residuals[masked_previously])))

    if max_vals and max(max_vals) < 1.5 * threshold:
        ax2.set_ylim(-1.5 * threshold, 1.5 * threshold)

    sm_res = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=[ax1,ax2], label="Cavity resonance  [GHz]")

    ax1.set_ylabel("PSD  [V²/Hz]")
    ax1.set_title(f"Spectra post cleaning - Iteration {iteration}")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.set_xlabel("Residuals [V²/Hz]")
    ax2.set_ylabel("PSD  [V²/Hz]")
    ax2.grid(alpha=0.3); ax2.legend()
    
    plt.savefig(run_dir / f"masked_bin_iteration_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close()