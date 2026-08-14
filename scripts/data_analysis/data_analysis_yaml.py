#!/usr/bin/env python
"""
Simulate a haloscope scan from a YAML config:
- simulation parameters
- optional axion injection
- optional saving of per-spectrum PNGs and spectra.npz
Outputs to: ./output/run_DD.MM.YYYY_HH.MM.SS
"""
from __future__ import annotations
import argparse
import datetime
import pathlib
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import matplotlib.cm as cm
from tqdm import tqdm


from axion_haloscope                      import graphs
from axion_haloscope.baseline             import remove_baseline
from axion_haloscope.combine              import combine_ml
from axion_haloscope.data_cuts            import cut_by_values, cut_by_datetime
from axion_haloscope.data_quality_working import filter_spectrum_set, too_noisy, power_too_high, metadata_is_zeros, time_filter, small_bandwidth
from axion_haloscope.detection            import threshold_for_detection, find_candidates
from axion_haloscope.diagnostics          import evaluate_set_spacing, vary_set_size_plots
from axion_haloscope.io_working           import write_hdf5
from axion_haloscope.limit                import compute_local_snr_template, coupling_limit
from axion_haloscope.lineshape            import shm_maxwell_template
from axion_haloscope.load_data            import load_data
from axion_haloscope.rebin                import rebin_ml, grand_spectrum_ml
from axion_haloscope.sets                 import set_creation, group_sets
from axion_haloscope.sigma_clipping       import claude_clipping, blue_clipping, finalise_specs, general_clipping
from axion_haloscope.utils                import create_directory, find_project_root, load_yaml_config

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman"],
    "font.size":   16,
})

def main():

    # ===================
    # Initialising
    # ===================

    ap = argparse.ArgumentParser(description="Simulate haloscope run from YAML config")
    ap.add_argument("config", help="Path to YAML config (e.g. configs/simulate_run.yaml)")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    cfg = load_yaml_config(cfg_path)
    inp, sim, inj, qc, alg, base, rb, det, out, diag = (cfg[k] for k in ("input","simulation","injection","quality",
                                                                         "alignment","baseline","rebin","detection","output","diagnostic"))
    # ----------------------------
    # Initialising Variables
    # ----------------------------

    '''
    Can replace all cfg["Settings"] with a global variable here
    '''


    # ----------------------------
    # Set Up Paths
    # ----------------------------

    # Output folder
    project_root = find_project_root(pathlib.Path(__file__).resolve())

    out_root = project_root / out["root"] / "data_analysis"
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    run_dir = create_directory(out_root, f'{out["subdir_prefix"]}_{timestamp}')
    main_plots_dir = create_directory(run_dir, "main_plots")
    data_dir = create_directory(run_dir, "data")

    if diag["run_diagnostics"]:
        print("Diagnostic Mode On")
        diagnostic_mode = True
        diag_run_dir = create_directory(run_dir, "diagnostic_plots")
    else:
        diagnostic_mode = False

    # ----------------------------
    # Save YAML
    # ----------------------------

    # Timestamped copies of the config
    cfg_stamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    try:
        # Save exact input YAML as provided
        stamped_name = f"{cfg_path.stem}_{cfg_stamp}{cfg_path.suffix}"
        shutil.copy(cfg_path, run_dir / stamped_name)
    except Exception as e:
        print(f"[WARN] Could not copy input config file: {e}")

    t_sim0 = time.time()
    
    
    
    # =======================================================================
    # Data input
    # =======================================================================
    sset = load_data(inp["input_mode"], diagnostic_mode, inp["directory"], inp["input_file_name"], data_dir, sim, inj, out)
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata
    initial_specs = specs

    # =======================================================================
    # Quality Control
    # =======================================================================

    if diagnostic_mode:
        print("=" * 60) 
        print("Quality Control")
        print("=" * 60)

        qc_run_dir = create_directory(diag_run_dir, "quality_control")

    invalid_files = sset.metadata.invalid_files


    bad_zero_power = []
    bad_no_metadata = []

    for f in invalid_files:
        if f[1] == "power spectra is zeros":
            bad_zero_power.append(f)
        elif f[1] == "metadata is missing":
            bad_no_metadata.append(f)
    if diagnostic_mode:
        print(f"[QC]: {len(bad_no_metadata)} spectra were removed as files were missing metadata.")
        print(f"[QC]: {len(bad_zero_power)} spectra were removed as power data were arrays of zeros.")

    step = max(1, int(out["plots_step"]))
    max_plots = None if out["max_plots"] is None else int(out["max_plots"])

    # QC: Cut spectra with power above threshold
    if qc["max_power_filter"]:
        sset, sset_power, kept, bad_power = filter_spectrum_set(sset,
            predicate=lambda s, f, md, i: power_too_high(s, f, md, i, p_max=qc["p_max"]))
        
        for idx, s in enumerate(bad_power):
            invalid_files.append([sset_power.metadata.file_names[idx], "power is too high", sset_power.metadata.dates[idx]])
        # Seperate invalid power spectra to plot
        specs_invalid_power, fper_invalid_power = sset_power.spectra, sset_power.freqs_per_spec
        if len(specs_invalid_power) != 0 and diagnostic_mode:
            print(f"[QC]: {len(bad_power)} spectra were removed as power is too high.")
            graphs.plot_spectrum(fper_invalid_power[0]/1e9, specs_invalid_power[0],
                        "Example invalid raw spectrum (high power)", qc_run_dir/"invalid_power_raw_spectrum_first.png")

            graphs.plot_spectrum(fper_invalid_power[-1]/1e9, specs_invalid_power[-1],
                        "Example invalid raw spectrum (high power)", qc_run_dir/"invalid_power_raw_spectrum_last.png")


    # QC: Cut spectra that have rms values above threshold
    if qc["noise_filter"]:
        sset, sset_noise, kept, bad_noise = filter_spectrum_set(sset,
            predicate=lambda s, f, md, i: too_noisy(s, f, md, i, rms_max=qc["rms_max"]))
        
        for idx, s in enumerate(bad_noise):
            invalid_files.append([sset_noise.metadata.file_names[idx], "too noisy", sset_noise.metadata.dates[idx]])
        # Seperate invalid noise spectra to plot
        specs_invalid_noise, fper_invalid_noise = sset_noise.spectra, sset_noise.freqs_per_spec
        if len(specs_invalid_noise) != 0 and diagnostic_mode:
            print(f"[QC]: {len(bad_noise)} spectra were removed as too noisy.")
            
            graphs.plot_spectrum(fper_invalid_noise[0]/1e9, specs_invalid_noise[0],
                        "Example invalid raw spectrum (too noisey)", qc_run_dir/"invalid_noise_raw_spectrum_first.png")

            graphs.plot_spectrum(fper_invalid_noise[-1]/1e9, specs_invalid_noise[-1],
                        "Example invalid raw spectrum (too noisey)", qc_run_dir/"invalid_noise_raw_spectrum_last.png")

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])


    # QC: Cut spectra that are within known bad times
    if qc["bad_time_filter"]:
        total_bad_time_filter = 0
        if len(qc["start_time"]) != len(qc["end_time"]):
            raise ValueError("The lists of start times and end times for cutting data are different lengths, please resolve this issue.")

        for t in range(len(qc["start_time"])):
            sset, sset_time_filtered, kept, bad_time_filter = filter_spectrum_set(sset,
                predicate=lambda s, f, md, i: time_filter(s, f, md, i, start_time = qc["start_time"][t], end_time = qc["end_time"][t]))
            total_bad_time_filter += len(bad_time_filter)
            for idx, s in enumerate(bad_time_filter):
                invalid_files.append([sset_time_filtered.metadata.file_names[idx], 
                                      f"known bad data ({qc['start_time'][t]}-{qc['end_time'][t]})" , sset_time_filtered.metadata.dates[idx]])
            
            specs_invalid_time, fper_invalid_time = sset_time_filtered.spectra, sset_time_filtered.freqs_per_spec
            if len(specs_invalid_time) != 0 and diagnostic_mode:
                
                start_safe = str(qc['start_time'][t]).replace(":", "-").replace(" ", "_")
                end_safe = str(qc['end_time'][t]).replace(":", "-").replace(" ", "_")

                graphs.plot_spectrum(fper_invalid_time[0]/1e9, specs_invalid_time[0],
                            f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})",
                            qc_run_dir/f"invalid_time_raw_spectrum_first_{start_safe}-{end_safe}.png")

                graphs.plot_spectrum(fper_invalid_time[-1]/1e9, specs_invalid_time[-1],
                            f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})",
                            qc_run_dir/f"invalid_time_raw_spectrum_last_{start_safe}-{end_safe}.png")

                step = max(1, int(out["plots_step"]))
                max_plots = None if out["max_plots"] is None else int(out["max_plots"])
        if diagnostic_mode:    
            print(f"[QC]: {total_bad_time_filter} spectra were removed as within known bad data times.")


    # QC: Cut spectra that have bandwidth values below the threshold value
    if qc["small_bandwidth_filter"]:
        sset, sset_bandwidth, kept, bad_bandwidth = filter_spectrum_set(sset,
            predicate=lambda s, f, md, i: small_bandwidth(s, f, md, i, bw_min=qc["bw_min"]))
        
        for idx, s in enumerate(bad_bandwidth):
            invalid_files.append([sset_bandwidth.metadata.file_names[idx], "bandwidth is too small", sset_bandwidth.metadata.dates[idx]])
        # Seperate invalid bandwidth spectra to plot
        specs_invalid_bandwidth, fper_invalid_bandwidth = sset_bandwidth.spectra, sset_bandwidth.freqs_per_spec
        if len(specs_invalid_bandwidth) != 0 and diagnostic_mode:
            print(f"[QC]: {len(bad_bandwidth)} spectra were removed as bandwidth is too small.")
            
            graphs.plot_spectrum(fper_invalid_bandwidth[0]/1e9, specs_invalid_bandwidth[0],
                        f"Example invalid raw spectrum (invalid bandwidth)", qc_run_dir/f"invalid_bandwidth_spectrum_first.png")

            graphs.plot_spectrum(fper_invalid_bandwidth[-1]/1e9, specs_invalid_bandwidth[-1],
                                    f"Example invalid raw spectrum (invalid bandwidth)", qc_run_dir/f"invalid_bandwidth_spectrum_last.png")

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])
            

            # Plot bandwidth againist date to show which are below the threshold
            good_bandwidths = np.array(sset.metadata.bandwidths)
            good_dates = pd.to_datetime(sset.metadata.dates)
            good_order = np.argsort(good_dates)
            bad_bandwidths = np.array(sset_bandwidth.metadata.bandwidths)
            bad_dates = pd.to_datetime(sset_bandwidth.metadata.dates)

            order = np.argsort(bad_dates)
            bad_dates_sorted = bad_dates[order]
            bad_bandwidths_sorted = bad_bandwidths[order]

            graphs.plot_bandwidth(bad_dates_sorted, bad_bandwidths_sorted, good_dates, good_bandwidths, good_order, qc_run_dir, qc)


    # QC: Cut spectra that have a res_freq value of zero
    if qc["res_freq_zeros_filter"]:
        sset, sset_zeros_res_freq, kept, bad_zeros_res_freq = filter_spectrum_set(sset,
            predicate=lambda s, f, md, i: metadata_is_zeros(s, f, md, i, item = "res_freqs"))
        for idx, s in enumerate(bad_zeros_res_freq):
            invalid_files.append([sset_zeros_res_freq.metadata.file_names[idx], "res_freq data is zeros", sset_zeros_res_freq.metadata.dates[idx]])
        if len(bad_zeros_res_freq) != 0 and diagnostic_mode:
            print(f"[QC]: {len(bad_zeros_res_freq)} spectra were removed as given res_freq is zero.")


    # QC: Cut spectra that have no injected axion (cw_freq = 0)
    if qc["cw_freq_zeros_filter"]:
        no_inj_files = []
        sset, sset_zeros_cw_freq, kept, bad_zeros_cw_freq = filter_spectrum_set(sset,
            predicate=lambda s, f, md, i: metadata_is_zeros(s, f, md, i, item = "cw_freqs"))
        for idx, s in enumerate(bad_zeros_cw_freq):
            no_inj_files.append([sset_zeros_cw_freq.metadata.file_names[idx], sset_zeros_cw_freq.metadata.dates[idx]])
        if len(bad_zeros_cw_freq) != 0 and diagnostic_mode:
            print(f"[QC]: {len(bad_zeros_cw_freq)} spectra have no injected axion (cw_freq = 0)")


    print(f"[QC]: {len(kept)} / {len(kept) + len(invalid_files)} files are valid and suitable for anaylsis, {len(invalid_files)} files are invalid.")
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    if diagnostic_mode:
        raw_run_dir = create_directory(diag_run_dir, "raw_spectra_plots")

    # ----------------------------
    # Plotting of filtered data
    # ----------------------------

    # Plot histogram of data againist time
    if diagnostic_mode:
        time_cut_dir = create_directory(diag_run_dir, "time_cuts")
        graphs.plot_filtered_data(metadata, invalid_files, "pre_time_cut", time_cut_dir)
        graphs.plot_rms_against_time(sset, time_cut_dir)

    
    # Export the spectrum set of all valid files
    if out["save_data"]:
        out_h5 = f"{data_dir}/valid_converted_spectra.h5"
        write_hdf5(sset, out_h5)
        print(f"[OUT]: Valid files SpectrumSet saved to: {out_h5}")


    # =======================================================================
    # TIME CUTTING
    # =======================================================================

    # Imported from Claude's Code
    time_array = [
    ["2026-01-27 00:00:00", "2026-01-31 09:55:41"], #-10 run, folder is Jan
    ["2026-01-27 14:30:00", "2026-01-28 04:50:00"],
    ['2026-02-01 00:10:58', '2026-02-10 18:10:58'], #-20 run, change to Feb
    ['2026-02-01 00:10:58', '2026-02-05 19:10:58'], #one small jmup
    ['2026-02-05 00:10:58', '2026-02-05 19:10:58'], #full linear section
    ['2026-02-01 00:10:58', '2026-02-04 22:30:58'] #low freq linear section
    ]
    time_array_index = 1

    sset = cut_by_datetime(
        sset,
        time_array[time_array_index][0],
        time_array[time_array_index][1],
    )
    if diagnostic_mode:
        print("-" * 60)
        print("Applying Time Cuts")
        print("-" * 60)
        print("[TF]:", time_array[time_array_index][0], "-->", time_array[time_array_index][1])
    print(f"[TF]: {len(sset.metadata.file_names)} files kept after time filter "
       f"(removed {len(metadata.file_names) - len(sset.metadata.file_names)})")
    
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    cw_freqs = np.array(metadata.cw_freqs)
    res_freqs = np.array(metadata.res_freqs)

    out_h5 = f"{data_dir}/final_converted_spectra.h5"
    write_hdf5(sset, out_h5)
    print(f"[OUT]: Post Time Cut SpectrumSet saved to: {out_h5}")

    # ----------------------------
    # Plotting of time cut data
    # ----------------------------
    if diagnostic_mode:
        graphs.plot_filtered_data2(metadata, invalid_files, "post_time_cut", time_cut_dir)
   
    # =============================================================
    # Spectra Plotting
    # =============================================================
    step = max(1, int(out["plots_step"]))
    max_plots = None if out["max_plots"] is None else int(out["max_plots"])

    plot_count = 0
    if out["save_data"]:
        print(f"[OUT]: valid spectra.npz saved to: {data_dir}/spectra.npz")
        # Optional: spectra.npz for valid data
        np.savez(data_dir/"spectra.npz", spectra=np.array(specs), freqs=fper, rf_grid=rf)

    if diagnostic_mode:
        if diag["save_raw_plots"]:
            for i, (freq, spec) in tqdm(enumerate(zip(fper, specs)), total=len(fper), desc="[DIAG]: Generating Raw Plots"):
                if i % step != 0:
                    continue
                if max_plots is not None and plot_count >= max_plots:
                    break

                graphs.plot_spectrum(freq/1e9, spec, f"Spectrum {i:03d}", raw_run_dir / f"spectrum_{i:03d}.png")
                plot_count += 1
            print(f"[DIAG]: Saved raw plots to: {raw_run_dir}")

        # Always save one valid example raw spectrum
        graphs.plot_spectrum(fper[0]/1e9, specs[0], f"Example valid raw spectrum", qc_run_dir/f"valid_raw_spectrum_first.png")
        graphs.plot_spectrum(fper[-1]/1e9, specs[-1], f"Example valid raw spectrum", qc_run_dir/f"valid_raw_spectrum_last.png")

        # Optional: plot all valid/invalid raw spectra in one figure
        if diag["combined_plot"]:
            graphs.plot_spectra(fper, specs, plot_count, max_plots, raw_run_dir, step, "All valid raw spectra", "raw_valid_spectrum_all.png")

            if len(specs_invalid_power) != 0:
                graphs.plot_spectra(fper_invalid_power, specs_invalid_power, plot_count, max_plots,
                            raw_run_dir, step, "All invalid raw spectra", "raw_invalid_spectrum_all.png")


        # Optional: plot all valid raw spectra in one figure with offset
        if diag["offset_combined_plot"]:
            # Calculate difference in resonant frequency of the cavity between the spectra
            res_freq_diff = []
            for f in res_freqs:
                difference = f - res_freqs[0]
                res_freq_diff.append(difference)
            # Plot the resonance frequency offset againist the spectrum index
            graphs.plot_scatter(res_freq_diff, raw_run_dir)

            # Combine the offset spectra into one figure
            graphs.plot_spectra(fper, specs, plot_count, max_plots, raw_run_dir, step, "All valid spectra offset", "raw_spectrum_all_valid_offset.png", offset=res_freq_diff)


        # Plot injected frequency distrubtion (frequency againist time)
        if diag["injection_distribution"]:
            colour_vals = (cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz
            metadata_dates = pd.to_datetime(metadata.dates, format="%Y-%m-%d %H:%M:%S")
            graphs.plot_evo_of_freq(colour_vals, metadata_dates, r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]", raw_run_dir)

    t0 = time.time()


    # =======================================================================
    # SPECTRUM CUTS
    # =======================================================================

    sset = cut_by_values(sset, cut_min_val = -0.3e6, cut_max_val = 2.3e6)
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    if diagnostic_mode:
        cut_run_dir = create_directory(diag_run_dir, "cut_spectra")


        # Plot example trimmed spectra
        graphs.plot_spectrum(fper[0]/1e9, specs[0], "Example Spectrum (First) - Trimmed and Validated",
                    cut_run_dir/"trimmed_spectrum_first.png", label = "Trimmed Spectrum")
        graphs.plot_spectrum(fper[-1]/1e9, specs[-1], "Example Spectrum (Last) - Trimmed and Validated",
                    cut_run_dir/"trimmed_spectrum_last.png", label = "Trimmed Spectrum")


    # =======================================================================
    # GAIN ALIGNMENT
    # =======================================================================

    if alg["gain_alignment"]:
        # Average every spectra (local averages)
        psd_averages = []
        for spec in specs:
            av = np.nanmean(spec)
            psd_averages.append(av)

        # Average the averages (global average)
        global_average = np.nanmedian(psd_averages)

        # Calculate differences in global and local averages
        psd_differences = []
        for psd in psd_averages:
            difference = global_average - psd
            psd_differences.append(difference)

        # Apply the differences to each spectrum
        shifted_spectra = []
        for spec,diff in zip(specs, psd_differences):
            shift = spec + diff
            shifted_spectra.append(shift)

        # Reassign specs as the shifted spectrum
        specs = shifted_spectra

    # =======================================================================
    # Data Cleaning
    # =======================================================================
    if qc["data_cleaning"] and diagnostic_mode:
        data_clean_dir = create_directory(diag_run_dir, 'data_cleaning')

        n_iter = base["n_iterations"]
        new_specs = []
        for spec_idx, (freq, spec) in enumerate(zip(fper, specs)):
            mask = None
            for iteration in range(1, n_iter + 1):
                mask, baseline, residuals, threshold, _ = general_clipping(
                    spec, base["sg_window_warm"], base["sg_poly_warm"], base["sigma_cut"], freqs=freq,
                    current_mask=mask, iteration=iteration
                )
                unmasked = mask == 0
                masked_previously = (mask > 0) & (mask != iteration)
                masked_this_iteration = mask == iteration

                new_spec = spec.copy()
                new_spec[~unmasked] = baseline[~unmasked]
                   
                if diagnostic_mode and masked_this_iteration.any():
                    graphs.plot_data_cleaning(freq, spec, metadata, baseline, threshold, 
                                    residuals, spec_idx, masked_this_iteration, 
                                    masked_previously, mask, unmasked, iteration=iteration, base=base, run_dir=data_clean_dir)
            new_specs.append(new_spec)
        specs = new_specs

    # =======================================================================
    # Warm Baseline Removal
    # =======================================================================

    if diagnostic_mode:
        print("=" * 60)
        print("Warm Baseline Removal")
        print("=" * 60)

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------
    if diagnostic_mode:
        warm_run_dir = create_directory(diag_run_dir, 'warm_baseline')

    spacing_minutes = base["spacing_minutes"]
    dts = metadata.dates
    date_times=[]
    for dt in dts:
        try:
            date_time = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            date_times.append(date_time)
        except ValueError as e:
            print(f"{dt} -> {e}")

    # -----------------------------------------------------------------------
    # Creating sets
    # -----------------------------------------------------------------------

    sets, set_avg_spectra, set_sg_fits = set_creation(date_times, spacing_minutes, specs, fper, metadata, base)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    set_mean_res = np.array([
    np.nanmean([item[2] for item in set], axis=0)
    for set in sets
    ])
 
    if diag["set_average_diagnostics"] and diagnostic_mode:
        set_av_spec_dir = create_directory(warm_run_dir, "set_averaged_spectra")

        set_and_av_dir = create_directory(warm_run_dir, "set_and_average_spectra")

        std_vs_freq_dir = create_directory(warm_run_dir, "std_vs_freq")

        set_av_spec_errors_dir = create_directory(set_av_spec_dir, "errors")

        set_av_spec_errors_zoom_dir = create_directory(set_av_spec_dir, "errors_zoom")

        set_hist_psd_dir = create_directory(warm_run_dir, "histogram_of_set_psd")

        # Plot set averaged spectra for all sets on one axis
        graphs.plot_sets("sets", fper, specs, set_mean_res, "Mean Cavity Resonance", warm_run_dir, f"Set-averaged spectra — all sets (n = {len(sets)})",
                  "set_averaged_spectra_all.png", plt.cm.viridis, set_avg_spectra = set_avg_spectra, set_sg_fits=None, sets=sets)

        # Plot set average spectra for all sets 3x3
        graphs.plot_3x3("mean", sets, set_mean_res, "IF frequency  [MHz]", "PSD  [V²/Hz]", 
                 f"Set-averaged spectra — all sets (n = {len(sets)})", "set_averaged_spectra_all_3x3.png", warm_run_dir)

        # Plot standard deviation of averaged sets againist frequency for all sets 3x3
        graphs.plot_3x3("std", sets, set_mean_res, "Standard Deviation  [V²/Hz]", "IF frequency  [MHz]", 
                 f"Standard deviation of averaged spectra against frequency - all sets (n = {len(sets)})",
                 "std_vs_freq_all_3x3.png", warm_run_dir) 

        # Plot standard deviation of averaged sets againist frequency for all sets
        graphs.plot_std_freq(sets, set_mean_res, warm_run_dir)
    
        # Plot a histogram of standard deviation of averaged sets for all sets
        n = len(sets)
        graphs.plot_hist(data=[np.std([x[0] for x in set], axis=0) for set in sets],
                vline=None, n=n, bins=50, xlabel="Standard Deviation of average  [V²/Hz]", vlabel=None, 
                title=f"Histogram of standard deviation of averaged sets - all sets (n = {len(sets)})",
                cb_label="Mean cavity resonance [GHz]", output_loc=f"{warm_run_dir}/std_hist_all" )


        # Plot average std for each set againist set number
        av_stds = []
        for s, set in enumerate(sets):
            std = np.std([x[0] for x in set], axis=0)
            av_stds.append(np.mean(std))
        graphs.plot_std_set_num(av_stds, warm_run_dir)


        for s, set in enumerate(tqdm(sets, desc="[DIAG]: Set averaging diagnostic plots")):
        # for g, (freqs, specs) in enumerate(set_avg_spectra):

            # Plot set averaged spectra + the sets spectra per set
            graphs.plot_spectra_in_set(set, s, set_and_av_dir)

            # Plot set averaged spectra with errors per set
            graphs.plot_set_average_errors(set, s, set_av_spec_errors_dir)


            # Plot zoomed set averaged spectra with errors zoomed in per set
            graphs.plot_zoom_set_average_errors(set, s, set_av_spec_errors_zoom_dir)
        

            # Plot histogram of each set averaged spectra per set
            mean_val = np.mean([x[0] for x in set])
            med_val = np.median([x[0] for x in set])
            graphs.plot_hist(data=np.mean([x[0] for x in set], axis=0), vline=[mean_val, med_val],
                    n=1, bins=100, xlabel="PSD  [V²/Hz]", vlabel=["mean value", "median value"],
                    title=f"Histogram of set averaged set {s}", cb_label=None, output_loc=f"{set_hist_psd_dir}/histogram_{s}")


            # Plot standard deviation of each set average againist frequency per set
            graphs.plot_std_against_freq(set, s, set_mean_res, std_vs_freq_dir)

    if diagnostic_mode:
        graphs.plot_sets("sg_fit", fper, specs, set_mean_res, "Mean Cavity Resonance", warm_run_dir,
                "Set-averaged spectra with initial SG fits  (dashed = fit)", "set_averaged_spectra_with_sg_fits.png",
                cmap=plt.cm.viridis, set_avg_spectra=set_avg_spectra,set_sg_fits=set_sg_fits)


    # -----------------------------------------------------------------------
    # Iterative Sigma Clipping
    # -----------------------------------------------------------------------

    if base["clipping_mode"].lower() == "claude":
        set_masks = [
            np.zeros(len(avg[0]), dtype=int) if avg is not None else None
            for avg in set_avg_spectra
        ]
    elif base["clipping_mode"].lower() == "blue":
        set_masks = [
            [np.zeros(len(item[0]), dtype=int) for item in set]
            for set in sets
        ]
    else:
        raise ValueError(f"Clipping mode {base["clipping_mode"]} no found. Did you enter the correct name?")

    sigma_cut = base["sigma_cut"]
    n_iterations = base["n_iterations"]
    for iteration in range(1, n_iterations + 1):
        if diagnostic_mode:
        
            print(f"[WB] Iteration {iteration} / {n_iterations} ---")

        if base["clipping_mode"].lower() == "claude":
            set_masks, set_sg_fits = claude_clipping(
                set_avg_spectra, set_masks, set_sg_fits,
                sigma_cut, base["sg_window_warm"], base["sg_poly_warm"], iteration
                )
            plotting_set_masks = [
                [mask] * len(set) if mask is not None else None
                for mask, set in zip(set_masks, sets)
            ]

        elif base["clipping_mode"].lower() == "blue":
            set_masks, set_sg_fits = blue_clipping(
                sets, set_masks, set_sg_fits, sigma_cut,
                base["sg_window_warm"], base["sg_poly_warm"], iteration
                )
            plotting_set_masks = set_masks

        if diagnostic_mode:
            graphs.plot_iteritive_clipping(set_avg_spectra, plotting_set_masks, set_sg_fits,iteration, warm_run_dir, set_mean_res, set_masks)


    # --------------------
    # Residuals Plotting
    # --------------------

    if diag["clipping_residuals"] and diagnostic_mode:

        clip_run_dir = create_directory(diag_run_dir, f'clipping_plots_{base["clipping_mode"].lower()}')

        clip_hist_run_dir = create_directory(clip_run_dir, f'{base["clipping_mode"].lower()}_histogram_of_residuals')

        clip_residuial_run_dir = create_directory(clip_run_dir, f'{base["clipping_mode"].lower()}_residuals_from_clipping')
        

        for s, fit in enumerate(tqdm(set_sg_fits, desc="[DIAG]: Clipping residuals plots")):

            if base["clipping_mode"].lower() == "claude":

                avg = set_avg_spectra[s]
                if avg is None or fit is None:
                    continue
                freqs, specs = avg
                residuals = specs - fit

                # Plot residuals againist frequency
                graphs.plot_claude_residuals(freqs, residuals, s, clip_residuial_run_dir)

                # Plot histogram of residuals
                graphs.plot_hist(data=residuals[np.isfinite(residuals)], vline=None, n=1, bins=50, xlabel="IF frequency  [MHz]",
                        vlabel=None, title=f"Residuals - set {s} (Claude's clipping method)", cb_label=None, 
                        output_loc=f"{clip_hist_run_dir}/histogram_{s}.png")
        


            elif base["clipping_mode"].lower() == "blue":

                set = sets[s]
                if fit is None or len(set) == 0:
                    continue

                
                # Plot residuals againist frequnecy for each set
                all_residuals = graphs.plot_blue_residuals(set, fit, cm.viridis(np.linspace(0, 1, len(set))), s, clip_residuial_run_dir)


                # Plot stacked histogram of residuals in each set
                graphs.plot_hist(data=[r[np.isfinite(r)] for r in all_residuals],
                        vline=None, n=len(set), bins=50, xlabel="Residuals  [V²/Hz]", vlabel=None,
                        title=f"Residuals histogram (stacked) — set {s} (Blue's clipping method)", cb_label="Spectrum index in set", 
                        output_loc=f"{clip_hist_run_dir}/histogram_{s}.png")


    # --------------------
    # Varying Set Size
    # --------------------


    if diag["varying_set_size"] and diagnostic_mode:
        specs_set = shifted_spectra
        var_dir = create_directory(diag_run_dir, 'varying_set_size')

        spacings_config = [5, 10, 15, 30, 60, 90, 120, 150, 180, 210, 240]  # minutes
        sets_by_spacing = {sp: group_sets(date_times, sp, specs_set, fper, metadata) for sp in spacings_config}

        var_results = [
            evaluate_set_spacing(sp, sets_by_spacing[sp], base, sigma_cut, n_iterations)
            for sp in spacings_config
        ]

        vary_set_size_plots(var_results, spacings_config, sets_by_spacing, var_dir, base)

        print("\n[Set size variation summary]")
        for r in var_results:
            print(f"  set size={r['spacing_minutes']:>4} min | "
                f"n_sets={r['n_sets']:>4} | "
                f"average size={r['average_set_size']:.1f} | "
                f"average residual std={r['average_residual_std']:.4g} | "
                f"masked={r['total_masked']:>6}/{r['total_bins']:<6}")

    # --------------------
    # Final Baseline Removal
    # --------------------
    specs, fper = finalise_specs(base["clipping_mode"].lower(), set_avg_spectra, sets, set_sg_fits)

    colour_vals = np.abs(cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz


    graphs.plot_sets("baseline_removal", fper, specs, colour_vals, r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]", 
                    main_plots_dir, "Set-averaged spectra with initial SG fits  (dashed = fit)",
                    "warm_baseline_removal.png", cmap=plt.cm.inferno)

  
    # =======================================================================
    # "Cold" Baseline Removal
    # =======================================================================

    if diagnostic_mode:
        print("=" * 60)
        print("Cold Baseline Removal")
        print("=" * 60)

    _= remove_baseline(
    spectrum=specs[0],
    window_length=base["sg_window_cold"],
    polyorder=base["sg_poly_cold"],
    subtract_one=True,
    diagnostic={"outfile": main_plots_dir / "cold_baseline_removal.png",
                "title": "Baseline removal (spectrum 0)"},
    freqs_hz=fper[0],
    )

    proc = []
    for s in specs:
        processed, _baseline = remove_baseline(
            s,
            window_length=base["sg_window_cold"],
            polyorder=base["sg_poly_cold"],
            subtract_one=True,
        )
        proc.append(processed)

    rf_map_new = [i - i[0] for i in rf_map]

    # =======================================================================
    # Combination
    # =======================================================================
    combined, sigma_c, counts = combine_ml(proc, rf_map_new, total_rf_bins=len(rf))
    graphs.plot_combination(rf, combined, main_plots_dir)

    # =======================================================================
    # Rebin + Grand Spectrum (SHM template)
    # =======================================================================
    rebin_width, template_width = rb["C"], rb["K"]
    data_rebinned, sigma_dr, _ = rebin_ml(combined, sigma_c, rebin_width=rebin_width)
    feqs_rebinned = rf[:len(data_rebinned)*rebin_width:rebin_width] + (rebin_width//2)* (fper[0][1] - fper[0][0])
    f0 = feqs_rebinned[len(feqs_rebinned)//2]
    f0 = np.average(metadata.res_freqs) * 1e9

    lineshape_template = shm_maxwell_template(template_width=template_width, bin_width_hz=rebin_width*(fper[0][1] - fper[0][0]), f0_hz=f0)
    grand_spectrum, sigma_gs = grand_spectrum_ml(data_rebinned, sigma_dr, lineshape_template)

    z = np.zeros_like(grand_spectrum)
    m = np.isfinite(sigma_gs) & (sigma_gs>0)
    z[m] = grand_spectrum[m]/sigma_gs[m]
    graphs.plot_grand_spectrum(feqs_rebinned, z, main_plots_dir)

    # =======================================================================
    # Candidates
    # =======================================================================

    theta = threshold_for_detection(det["target_snr"], det["confidence"])
    if diagnostic_mode:
        print(f"[DETECT] Detection threshold theta = {theta:.3f} sigma "
              f"(target_snr={det['target_snr']}, confidence={det['confidence']})")
    cands, _ = find_candidates(grand_spectrum, sigma_gs, theta, min_separation=template_width-1)
    if diagnostic_mode:
        print(f"[DETECT] Found {len(cands)} candidate(s) above threshold")
    
    graphs.plot_candidates(feqs_rebinned, z, theta, cands, main_plots_dir)


    t1     = time.time()
    total0 = round(t1-t0, 2)
    totals = round(t0-t_sim0, 2)
    if inp["input_mode"] == "simulation":
        nbins    = sim["n_bins"]
        nspectra = sim["n_spectra"]
        print (f"[FOX]: Simulation Time : {totals} s for {nspectra} spectra of {nbins} bins")
    else:
        print (f"[FOX]: Data Loading Time : {totals} s for {len(metadata.file_names)} spectra of {len(initial_specs[0])} bins")
    print (f"[FOX]: Time from QC to Candidates: {total0} s")

    # =======================================================================
    # Exclusion
    # =======================================================================

    local_snr = compute_local_snr_template(sigma_dr, lineshape_template)
    gmin = coupling_limit(local_snr, target_snr=det["target_snr"], g0=det["g0"], snr_efficiency=det["snr_eff"])
    if diagnostic_mode:
        print(f"[EXCLUSION] Computing local SNR template and coupling limit (target_snr={det['target_snr']}, g0={det['g0']}, snr_eff={det['snr_eff']})")
        finite_g = gmin[np.isfinite(gmin)]
        if finite_g.size:
            print(f"[EXCLUSION] g_min (rel. to g0) stats: best={np.min(finite_g):.4g}, "
                    f"worst={np.max(finite_g):.4g}")
    graphs.plot_exclusion(feqs_rebinned, gmin, outfile=main_plots_dir/"exclusion.png", title="95% CL Exclusion (SHM)")
    with (data_dir/"exclusion.csv").open("w") as fh:
        fh.write("freq_Hz,g_min_rel_to_g0\n")
        for f,g in zip(feqs_rebinned, gmin):
            if np.isfinite(g): fh.write(f"{f},{g}\n")
    if diagnostic_mode:
        print(f"[OUT]: Exclusion CSV saved to: {data_dir/'exclusion.csv'}")

    print(f"[OUT]: Run dir: {run_dir}")
    print(f"[FOX]: Candidates flagged: {len(cands)}  (threshold = {theta:.2f}σ)")


if __name__ == "__main__":
    main()
