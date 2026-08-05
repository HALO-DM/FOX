#!/usr/bin/env python
"""
Simulate a haloscope scan from a YAML config:
- simulation parameters
- optional axion injection
- optional saving of per-spectrum PNGs and spectra.npz
Outputs to: ./output/run_DD.MM.YYYY_HH.MM.SS
"""
from __future__ import annotations
import argparse, datetime, pathlib, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import pandas as pd
import shutil
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm


from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape  import shm_maxwell_template
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit
from axion_haloscope.data_quality_working import filter_spectrum_set, too_noisy, power_too_high, metadata_is_zeros, time_filter, small_bandwidth
from axion_haloscope.io_working import SpectrumSet, read_hdf5, write_hdf5
from axion_haloscope.sigma_clipping import claude_clipping, blue_clipping, finalise_specs, general_clipping
from axion_haloscope.graphs import (plot_spectrum, vs_time_hist, plot_hist, plot_bandwidth, plot_events_against_time, 
                                   plot_rms_against_time, plot_spectra, plot_exclusion, plot_scatter, plot_evo_of_freq,
                                   plot_sets,plot_iteritive_clipping,plot_3x3, plot_std_freq, plot_std_set_num, 
                                   plot_spectra_in_set, plot_log_spectra_in_set, plot_set_average_errors, 
                                   plot_zoom_set_average_errors, plot_std_against_freq, plot_claude_residuals,
                                   plot_blue_residuals, plot_combination, plot_grand_spectrum, plot_candidates,plot_data_cleaning)
from axion_haloscope.sets import set_creation, group_sets
from axion_haloscope.diagnostics import evaluate_set_spacing, vary_set_size_plots
from axion_haloscope.data_cuts import cut_by_values, cut_by_datetime

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman"],
    "font.size":   16,
})

def _get(d, key, default):
    v = d.get(key, default)
    return default if v is None else v

def load_yaml_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    sim  = raw.get("simulation", {}) or {}
    inp  = raw.get("input",      {}) or {}
    inj  = raw.get("injection",  {}) or {}
    qc   = raw.get("quality",    {}) or {}
    base = raw.get("baseline",   {}) or {}
    rb   = raw.get("rebin",      {}) or {}
    det  = raw.get("detection",  {}) or {}
    out  = raw.get("output",     {}) or {}

    cfg = {
        "simulation": {
            "n_spectra":      int(_get(sim, "n_spectra", 80)),
            "n_bins":         int(_get(sim, "n_bins", 8000)),
            "bin_width_hz":   float(_get(sim, "bin_width_hz", 100.0)),
            "f_start_hz":     float(_get(sim, "f_start_hz", 5.70e9)),
            "tune_step_bins": int(_get(sim, "tune_step_bins", 100)),
            "rng_seed":       int(_get(sim, "rng_seed", 1234)),
            "noise_sigma":    float(_get(sim, "noise_sigma", 1.0)),
        },
        "input": {
            "read_input":           bool(_get(inp, "read_input", False)),
            "directory":            _get(inp, "directory", "scripts/qshs/output/qshs_import"),
            "input_file_name":      _get(inp, "input_file_name", "spectra.h5"),
            "full_data_directory":  _get(inp, "full_data_directory", "input/Feb/All"),
        },
        "injection": {
            "enabled":     bool(_get(inj, "enabled", False)),
            "f_axion_hz":  inj.get("f_axion_hz", None),  # optional
            "total_power": float(_get(inj, "total_power", 20.0)),
        },
        "quality": {
            "max_power_filter":         bool(_get(qc, "max_power_filter", True)),
            "p_max":                    float(_get(qc, "p_max", 1e-8)),
            "noise_filter":             bool(_get(qc, "noise_filter", True)),
            "rms_max":                  float(_get(qc, "rms_max", 1e-10)),
            "nan_fail":                 bool(_get(qc, "nan_fail", True)),
            "robust":                   bool(_get(qc, "robust", True)), 
            "small_bandwidth_filter":   bool(_get(qc, "small_bandwidth_filter", True)),
            "bw_min":                   float(_get(qc, "bw_min", 0.00027)),
            "bandwidth_zeros_filter":   bool(_get(qc, "bandwidth_zeros_filter", True)),
            "res_freq_zeros_filter":    bool(_get(qc, "res_freq_zeros_filter", True)),
            "cw_freq_zeros_filter":     bool(_get(qc, "cw_freq_zeros_filter", True)),
            "bad_time_filter":          bool(_get(qc, "bad_time_filter", True)),
            "start_time":               _get(qc, "start_time", None),
            "end_time":                 _get(qc, "end_time", None),
            "data_cleaning":            bool(_get(qc, "data_cleaning", True)),
        },
        "baseline": {
            "sg_window_warm": int(_get(base, "sg_window_warm", 251)),
            "sg_poly_warm":   int(_get(base, "sg_poly_warm", 2)),
            "sg_window_cold": int(_get(base, "sg_window_cold", 401)),
            "sg_poly_cold":   int(_get(base, "sg_poly_cold", 4)),
            "spacing_minutes":float(_get(base, "spacing_minutes", 30)),
            "sigma_cut":      float(_get(base, "sigma_cut", 3.5)),
            "clipping_mode":  _get(base, "clipping_mode", "Claude"),
            "n_iterations":   int(_get(base, "n_iterations", 3)),
        },
        "rebin": {
            "C": int(_get(rb, "C", 10)),
            "K": int(_get(rb, "K", 9)),
        },
        "detection": {
            "target_snr": float(_get(det, "target_snr", 5.0)),
            "confidence": float(_get(det, "confidence", 0.95)),
            "snr_eff":    float(_get(det, "snr_eff", 0.90)),
            "g0":         float(_get(det, "g0", 1.0)),
        },
        "output": {
            "save_data":                bool(_get(out, "save_data", False)),
            "combined_plot":            bool(_get(out, "combined_plot", False)),
            "offset_combined_plot":     bool(_get(out, "offset_combined_plot", False)),
            "injection_distribution":   bool(_get(out, "injection_distribution", False)),
            "set_average_diagnostics":  bool(_get(out, "set_average_diagnostics", False)),
            "clipping_residuals":       bool(_get(out, "clipping_residuals", False)),
            "varying_set_size":         bool(_get(out, "varying_set_size", False)),
            "plots_step":               int(_get(out, "plots_step", 1)),   # plot every Nth spectrum
            "max_plots":                out.get("max_plots", None),        # optional int
            "root":                     _get(out, "root", "output"),
            "subdir_prefix":            _get(out, "subdir_prefix", "run"),
        },
    }
    return cfg

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
    sim, inp, inj, qc, base, rb, det, out = (cfg[k] for k in ("simulation","input","injection","quality","baseline","rebin","detection","output"))

    # Output folder
    out_root = pathlib.Path(out["root"])/ "sim_spectra"
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped copies of the config
    cfg_stamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    # Save exact input YAML as provided
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        stamped_name = f"{cfg_path.stem}_{cfg_stamp}{cfg_path.suffix}"
        shutil.copy(cfg_path, data_dir / stamped_name)
    except Exception as e:
        print(f"[WARN] Could not copy input config file: {e}")

    t_sim0 = time.time()
    
    
    # =======================================================================
    # Data input
    # =======================================================================


    if inp["read_input"]:
        # 1) Read in Data
        directory = inp["directory"]
        input_file_name = inp["input_file_name"]
        sset = read_hdf5(f"{directory}/{input_file_name}")
        specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata
        initial_specs = specs
    else:

        # 1) Simulate
        specs, fper, rf, rf_map, metadata = simulate_spectra(
        n_spectra=sim["n_spectra"], n_bins=sim["n_bins"],
        bin_width_hz=sim["bin_width_hz"], f_start_hz=sim["f_start_hz"],
        tune_step_bins=sim["tune_step_bins"], rng_seed=sim["rng_seed"],
        noise_sigma=sim["noise_sigma"], injected_axion=inj
        )


    # =======================================================================
    # Quality Control
    # =======================================================================

    print("=" * 60)
    print(f"Quality Control")
    print("=" * 60)

    qc_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'quality_control'
    qc_run_dir.mkdir(parents=True, exist_ok=True)

    invalid_files = sset.metadata["invalid_files"]
    bad_zero_power = []
    bad_no_metadata = []

    for f in invalid_files:
        if f[1] == "power spectra is zeros":
            bad_zero_power.append(f)
        elif f[1] == "metadata is missing":
            bad_no_metadata.append(f)

    print(f"[QC]: {len(bad_no_metadata)} spectra were removed as files were missing metadata.")
    print(f"[QC]: {len(bad_zero_power)} spectra were removed as power data were arrays of zeros.")


    # QC: Cut spectra with power above threshold
    if qc["max_power_filter"]:
        sset, sset_power, kept, bad_power = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: power_too_high(
                s,
                f,
                md,
                i,
                p_max=qc["p_max"],
            ),
        )
        for idx, s in enumerate(bad_power):
            invalid_files.append([sset_power.metadata["file_name"][idx], "power is too high", sset_power.metadata["date"][idx]])
        print(f"[QC]: {len(bad_power)} spectra were removed as power is too high.")
        # Seperate invalid power spectra to plot
        specs_invalid_power, fper_invalid_power = sset_power.spectra, sset_power.freqs_per_spec
        if len(specs_invalid_power) != 0:
            plot_spectrum(fper_invalid_power[0]/1e9, specs_invalid_power[0],
                        "Example invalid raw spectrum (high power)", qc_run_dir/"invalid_power_raw_spectrum_first.png")

            plot_spectrum(fper_invalid_power[-1]/1e9, specs_invalid_power[-1],
                        "Example invalid raw spectrum (high power)", qc_run_dir/"invalid_power_raw_spectrum_last.png")

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])
            


    # QC: Cut spectra that have rms values above threshold
    if qc["noise_filter"]:
        sset, sset_noise, kept, bad_noise = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: too_noisy(
                s,
                f,
                md,
                i,
                rms_max=qc["rms_max"], 
            )
        )
        for idx, s in enumerate(bad_noise):
            invalid_files.append([sset_noise.metadata["file_name"][idx], "too noisy", sset_noise.metadata["date"][idx]])
        print(f"[QC]: {len(bad_noise)} spectra were removed as too noisy.")
        # Seperate invalid noise spectra to plot
        specs_invalid_noise, fper_invalid_noise = sset_noise.spectra, sset_noise.freqs_per_spec
        if len(specs_invalid_noise) != 0:
            
            plot_spectrum(fper_invalid_noise[0]/1e9, specs_invalid_noise[0],
                        "Example invalid raw spectrum (too noisey)", qc_run_dir/"invalid_noise_raw_spectrum_first.png")

            plot_spectrum(fper_invalid_noise[-1]/1e9, specs_invalid_noise[-1],
                        "Example invalid raw spectrum (too noisey)", qc_run_dir/"invalid_noise_raw_spectrum_last.png")

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])


    # QC: Cut spectra that are within known bad times
    if qc["bad_time_filter"]:
        total_bad_time_filter = 0
        if len(qc["start_time"]) != len(qc["end_time"]):
            raise ValueError("The lists of start times and end times for cutting data are different lengths, please resolve this issue.")
        
        else:
            for t in range(len(qc["start_time"])):
                sset, sset_time_filtered, kept, bad_time_filter = filter_spectrum_set(
                        sset,
                        predicate=lambda s, f, md, i: time_filter(
                            s,
                            f,
                            md,
                            i,
                            start_time = qc["start_time"][t],
                            end_time = qc["end_time"][t],
                        ),
                )
                total_bad_time_filter += len(bad_time_filter)
                for idx, s in enumerate(bad_time_filter):
                    invalid_files.append([sset_time_filtered.metadata["file_name"][idx], f"known bad data ({qc['start_time'][t]}-{qc['end_time'][t]})" , sset_time_filtered.metadata["date"][idx]])
                
                specs_invalid_time, fper_invalid_time = sset_time_filtered.spectra, sset_time_filtered.freqs_per_spec
                if len(specs_invalid_time) != 0:
                    
                    plot_spectrum(fper_invalid_time[0]/1e9, specs_invalid_time[0],
                                f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})",
                                qc_run_dir/f"invalid_time_raw_spectrum_first_{qc['start_time'][t]}-{qc['end_time'][t]}.png")

                    plot_spectrum(fper_invalid_time[-1]/1e9, specs_invalid_time[-1],
                                f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})",
                                qc_run_dir/f"invalid_time_raw_spectrum_last_{qc['start_time'][t]}-{qc['end_time'][t]}.png")

                    step = max(1, int(out["plots_step"]))
                    max_plots = None if out["max_plots"] is None else int(out["max_plots"])
                    
            print(f"[QC]: {total_bad_time_filter} spectra were removed as within known bad data times.")


    # QC: Cut spectra that have bandwidth values below the threshold value
    if qc["small_bandwidth_filter"]:
        sset, sset_bandwidth, kept, bad_bandwidth = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: small_bandwidth(
                s,
                f,
                md,
                i,
                bw_min=qc["bw_min"],
            ),
        )
        for idx, s in enumerate(bad_bandwidth):
            invalid_files.append([sset_bandwidth.metadata["file_name"][idx], "bandwidth is too small", sset_bandwidth.metadata["date"][idx]])
        print(f"[QC]: {len(bad_bandwidth)} spectra were removed as bandwidth is too small.")
        # Seperate invalid bandwidth spectra to plot
        specs_invalid_bandwidth, fper_invalid_bandwidth = sset_bandwidth.spectra, sset_bandwidth.freqs_per_spec
        if len(specs_invalid_bandwidth) != 0:
            
            plot_spectrum(fper_invalid_bandwidth[0]/1e9, specs_invalid_bandwidth[0],
                        f"Example invalid raw spectrum (invalid bandwidth)", qc_run_dir/f"invalid_bandwidth_spectrum_first.png")

            plot_spectrum(fper_invalid_bandwidth[-1]/1e9, specs_invalid_bandwidth[-1],
                                    f"Example invalid raw spectrum (invalid bandwidth)", qc_run_dir/f"invalid_bandwidth_spectrum_last.png")

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])
            

            # Plot bandwidth againist date to show which are below the threshold
            good_bandwidths = np.array(sset.metadata["bandwidth"])
            good_dates = pd.to_datetime(sset.metadata["date"])
            good_order = np.argsort(good_dates)
            bad_bandwidths = np.array(sset_bandwidth.metadata["bandwidth"])
            bad_dates = pd.to_datetime(sset_bandwidth.metadata["date"])

            order = np.argsort(bad_dates)
            bad_dates_sorted = bad_dates[order]
            bad_bandwidths_sorted = bad_bandwidths[order]

            plot_bandwidth(bad_dates_sorted, bad_bandwidths_sorted, good_dates, good_bandwidths, good_order, qc_run_dir, qc)


    # QC: Cut spectra that have a res_freq value of zero
    if qc["res_freq_zeros_filter"]:
        sset, sset_zeros_res_freq, kept, bad_zeros_res_freq = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: metadata_is_zeros(
                s,
                f,
                md,
                i,
                item = "res_freq",
            ),
        )
        for idx, s in enumerate(bad_zeros_res_freq):
            invalid_files.append([sset_zeros_res_freq.metadata["file_name"][idx], "res_freq data is zeros", sset_zeros_res_freq.metadata["date"][idx]])
        print(f"[QC]: {len(bad_zeros_res_freq)} spectra were removed as given res_freq is zero.")


    # QC: Flag spectra that have no injected axion (cw_freq = 0)
    no_inj_files = []
    if qc["cw_freq_zeros_filter"]:
        sset, sset_zeros_cw_freq, kept, bad_zeros_cw_freq = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: metadata_is_zeros(
                s,
                f,
                md,
                i,
                item = "cw_freq",
            ),
        )
        for idx, s in enumerate(bad_zeros_cw_freq):
            no_inj_files.append([sset_zeros_cw_freq.metadata["file_name"][idx], sset_zeros_cw_freq.metadata["date"][idx]])
        print(f"[QC]: {len(bad_zeros_cw_freq)} spectra have no injected axion (cw_freq = 0)")


    print(f"[QC]: {len(kept)} / {len(kept) + len(invalid_files)} files are valid and suitable for anaylsis, {len(invalid_files)} files are invalid.")
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    raw_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}'/ 'raw_spectra_plots'
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Plotting of filtered data
    # ----------------------------

    # Plot histogram of data againist time
    valid_files_df = pd.DataFrame(zip(metadata["file_name"] , metadata["date"]))
    valid_files_df[1] = pd.to_datetime(valid_files_df[1], format="%Y-%m-%d %H:%M:%S")

    if len(invalid_files) != 0:
        invalid_files_df = pd.DataFrame(invalid_files)
        invalid_files_df[2] = pd.to_datetime(invalid_files_df[2], format="%Y-%m-%d %H:%M:%S")
        invalid_metadata = invalid_files_df[invalid_files_df[1] == "metadata is missing"]
        invalid_high_power = invalid_files_df[invalid_files_df[1] == "power is too high"]
        invalid_high_noise = invalid_files_df[invalid_files_df[1] == "too noisy"]
        invalid_power_zeros = invalid_files_df[invalid_files_df[1] == "power spectra is zeros"]
        invalid_bandwidth = invalid_files_df[invalid_files_df[1] == "bandwidth is too small"]
        invalid_res_freq_zeros = invalid_files_df[invalid_files_df[1] == "res_freq data is zeros"]
        invalid_cw_freq_zeros = invalid_files_df[invalid_files_df[1] == "cw_freq data is zeros"]

        start_date = min( valid_files_df[1].min(), invalid_files_df[2].min())
        end_date = max( valid_files_df[1].max(), invalid_files_df[2].max())
        time_interval = pd.Timedelta(hours=24)
        bin_num = int( (end_date - start_date) / time_interval)

        data = [valid_files_df[1], invalid_metadata[2], invalid_high_power[2], invalid_high_noise[2], invalid_power_zeros[2], invalid_bandwidth[2], invalid_res_freq_zeros[2]]
        labels = [f"valid files (n = {len(valid_files_df[1])})", f"missing metadata (n = {len(invalid_metadata[2])})", f"power too high (n = {len(invalid_high_power[2])})", f"too noisy (n = {len(invalid_high_noise[2])})",
                        f"power is zeros (n = {len(invalid_power_zeros[2])})", f"bandwidth is too small (n = {len(invalid_bandwidth[2])})", f"res freq is zeros (n = {len(invalid_res_freq_zeros[2])})"]

        vs_time_hist( data, bin_num, (start_date, end_date), labels, "Number of files",
                    f"Spectra files per day (before timestamp filter) n = {len(valid_files_df[0] + invalid_files_df[0])}", raw_run_dir/"spectra_hist.png")


    else:
        start_date = valid_files_df[1].min()
        end_date = valid_files_df[1].max()
        time_interval = pd.Timedelta(hours=0.1)
        bin_num = int( (end_date - start_date) / time_interval)

        vs_time_hist(valid_files_df[1], bin_num, (start_date, end_date), f"valid files (n = {len(valid_files_df[1])})", "Number of files", 
                     "Spectra files per day (before timestamp filter)", raw_run_dir/"spectra_hist.png")


    if len(invalid_files) != 0:
        invalid_files_df = invalid_files_df.sort_values(by=[2])
        count_invalid = list(range(1, len(invalid_files_df[2]) + 1))
        all_files_df = pd.concat([valid_files_df[1], invalid_files_df[2]])
        all_files_df = all_files_df.sort_values()
        overall_end = max(valid_files_df[1].max(), invalid_files_df[2].max())

        invalid_dates = list(invalid_files_df[2])
        if invalid_dates[-1] < overall_end:
            invalid_dates.append(overall_end)
            count_invalid.append(count_invalid[-1])

    else:
        all_files_df = valid_files_df[1]
        all_files_df = all_files_df.sort_values()

    valid_files_df = valid_files_df.sort_values(by=[1])
    count_valid = list(range(1, len(valid_files_df[1]) + 1))
    count_all = list(range(1, len(all_files_df) + 1))

    overall_end = max(valid_files_df[1].max(), all_files_df.max())

    valid_dates = list(valid_files_df[1])
    if valid_dates[-1] < overall_end:
        valid_dates.append(overall_end)
        count_valid.append(count_valid[-1])

    all_dates = list(all_files_df)
    if all_dates[-1] < overall_end:
        all_dates.append(overall_end)
        count_all.append(count_all[-1])

    plot_events_against_time(invalid_dates, valid_dates, all_dates, count_invalid, count_valid, count_all, raw_run_dir)
    plot_rms_against_time(sset, qc_run_dir)

    
    # Export the spectrum set of all valid files
    out_h5 = f"{data_dir}/valid_converted_spectra.h5"
    write_hdf5(sset, out_h5)
    print(f"[QSHS] Valid files SpectrumSet saved to: {out_h5}")


    # =======================================================================
    # TIME CUTTING
    # =======================================================================

    n_before = len(metadata["file_name"])

    TIME_ARR = [
    ["2026-01-27 00:00:00", "2026-01-31 09:55:41"], #-10 run, folder is Jan
    ["2026-01-27 14:30:00", "2026-01-28 04:50:00"],
    ['2026-02-01 00:10:58', '2026-02-10 18:10:58'], #-20 run, change to Feb
    ['2026-02-01 00:10:58', '2026-02-05 19:10:58'], #one small jmup
    ['2026-02-05 00:10:58', '2026-02-05 19:10:58'], #full linear section
    ['2026-02-01 00:10:58', '2026-02-04 22:30:58'] #low freq linear section
    ]
    TIME_IND = 1

    sset = cut_by_datetime(
        sset,
        TIME_ARR[TIME_IND][0],
        TIME_ARR[TIME_IND][1],
    )

    print("[TF]:", TIME_ARR[TIME_IND][0], "-->", TIME_ARR[TIME_IND][1])
    print(f"[TF]: {len(sset.metadata['file_name'])} files kept after time filter "
       f"(removed {n_before - len(sset.metadata['file_name'])})")
    
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    cw_freqs = np.array(metadata["cw_freq"])
    res_freqs = np.array(metadata["res_freq"])

    out_h5 = f"{data_dir}/final_converted_spectra.h5"
    write_hdf5(sset, out_h5)
    print(f"[QSHS] Final SpectrumSet saved to: {out_h5}")

    # ----------------------------
    # Plotting of time cut data
    # ----------------------------

    # Plot histogram of data againist time (post time filter)
    invalid_files = metadata["invalid_files"]
    invalid_files_df = pd.DataFrame(invalid_files)
    valid_files_df = pd.DataFrame(zip(metadata["file_name"] , metadata["date"]))
    valid_files_df[1] = pd.to_datetime(valid_files_df[1], format="%Y-%m-%d %H:%M:%S")

    if len(invalid_files) != 0:
        invalid_files_df = pd.DataFrame(invalid_files)
        invalid_files_df[2] = pd.to_datetime(invalid_files_df[2], format="%Y-%m-%d %H:%M:%S")
        invalid_metadata = invalid_files_df[invalid_files_df[1] == "metadata is missing"]
        invalid_high_power = invalid_files_df[invalid_files_df[1] == "power is too high"]
        invalid_high_noise = invalid_files_df[invalid_files_df[1] == "too noisy"]
        invalid_power_zeros = invalid_files_df[invalid_files_df[1] == "power spectra is zeros"]
        invalid_bandwidth = invalid_files_df[invalid_files_df[1] == "bandwidth is too small"]
        invalid_res_freq_zeros = invalid_files_df[invalid_files_df[1] == "res_freq data is zeros"]
        invalid_time = invalid_files_df[invalid_files_df[1] == "not in good time range"]

        start_date = min( valid_files_df[1].min(), invalid_files_df[2].min())
        end_date = max( valid_files_df[1].max(), invalid_files_df[2].max())
        time_interval = pd.Timedelta(hours=24)
        bin_num = int( (end_date - start_date) / time_interval)


        data = (valid_files_df[1], invalid_metadata[2], invalid_high_power[2], invalid_high_noise[2], invalid_power_zeros[2], invalid_bandwidth[2], invalid_res_freq_zeros[2], invalid_time[2])
        labels = [ f"valid files (n = {len(valid_files_df[1])})", f"missing metadata (n = {len(invalid_metadata[2])})",
                f"power too high (n = {len(invalid_high_power[2])})", f"too noisy (n = {len(invalid_high_noise[2])})",
                f"power is zeros (n = {len(invalid_power_zeros[2])})", f"bandwidth is too small (n = {len(invalid_bandwidth[2])})",
                f"res freq is zeros (n = {len(invalid_res_freq_zeros[2])})", f"invalid time filter (n = {len(invalid_time)})"]
        vs_time_hist(data, bin_num, (start_date, end_date), labels, "Number of files", f"Spectra files per day (post timestamp filter) n = {len(invalid_files_df[0] + valid_files_df[0])}", raw_run_dir/"spectra_hist_time_cut.png")

    else:
        start_date = valid_files_df[1].min()
        end_date = valid_files_df[1].max()
        time_interval = pd.Timedelta(hours=0.1)
        bin_num = int( (end_date - start_date) / time_interval)

        vs_time_hist(valid_files_df[1], bin_num, (start_date, end_date), f"valid files (n = {len(valid_files_df[1])})", "Number of files", f"Spectra files per day (post timestamp filter) n = {len(valid_files_df[0])}", raw_run_dir/"spectra_hist_time_cut.png")


    # Plot of total number of events againist time (post time filter)
    if len(invalid_files) != 0:
        invalid_files_df = invalid_files_df.sort_values(by=[2])
        count_invalid = list(range(1, len(invalid_files_df[2]) + 1))
        all_files_df = pd.concat([valid_files_df[1], invalid_files_df[2]])
        all_files_df = all_files_df.sort_values()
        overall_end = max(valid_files_df[1].max(), invalid_files_df[2].max())

        invalid_dates = list(invalid_files_df[2])
        if invalid_dates[-1] < overall_end:
            invalid_dates.append(overall_end)
            count_invalid.append(count_invalid[-1])

    else:
        all_files_df = valid_files_df[1]
        all_files_df = all_files_df.sort_values()

    valid_files_df = valid_files_df.sort_values(by=[1])
    count_valid = list(range(1, len(valid_files_df[1]) + 1))
    count_all = list(range(1, len(all_files_df) + 1))

    overall_end = max(valid_files_df[1].max(), all_files_df.max())

    valid_dates = list(valid_files_df[1])
    if valid_dates[-1] < overall_end:
        valid_dates.append(overall_end)
        count_valid.append(count_valid[-1])

    all_dates = list(all_files_df)
    if all_dates[-1] < overall_end:
        all_dates.append(overall_end)
        count_all.append(count_all[-1])

    plot_events_against_time(invalid_dates, valid_dates, all_dates, count_invalid, count_valid, count_all, raw_run_dir)

    
    # =============================================================
    # Spectra Plotting
    # =============================================================
    step = max(1, int(out["plots_step"]))
    max_plots = None if out["max_plots"] is None else int(out["max_plots"])

    # Calculate difference in resonant frequency of the cavity between the spectra
    res_freq_diff = []
    for f in res_freqs:
        difference = f - res_freqs[0]
        res_freq_diff.append(difference)

    # Optional: save per-spectrum PNGs + spectra.npz for valid data
    count = 0
    if out["save_data"]:
        for i, (freq, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break

            plot_spectrum(freq/1e9, spec, f"Spectrum {i:03d}", raw_run_dir / f"spectrum_{i:03d}.png")
            count += 1
        np.savez(run_dir/"spectra.npz", spectra=np.array(specs), freqs=fper, rf_grid=rf)

    # Always save one valid example raw spectrum
    plot_spectrum(fper[0]/1e9, specs[0], f"Example valid raw spectrum", qc_run_dir/f"valid_raw_spectrum_first.png")
    plot_spectrum(fper[-1]/1e9, specs[-1], f"Example valid raw spectrum", qc_run_dir/f"valid_raw_spectrum_last.png")

    # Optional: plot all valid/invalid raw spectra in one figure
    if out["combined_plot"]:
        plot_spectra(fper, specs, count, max_plots, raw_run_dir, step, "All valid raw spectra", "raw_valid_spectrum_all.png")

        if len(specs_invalid_power) != 0:
            plot_spectra(fper_invalid_power, specs_invalid_power, count, max_plots, raw_run_dir, step, "All invalid raw spectra", "raw_invalid_spectrum_all.png")


    # Optional: plot all valid raw spectra in one figure with offset
    if out["offset_combined_plot"]:
        # Plot the resonance frequency offset againist the spectrum index
        plot_scatter(res_freq_diff, raw_run_dir)

        # Combine the offset spectra into one figure
        plot_spectra(fper, specs, count, max_plots, raw_run_dir, step, "All valid spectra offset", "raw_spectrum_all_valid_offset.png", offset=res_freq_diff)


    # Plot injected frequency distrubtion (frequency againist time)
    if out["injection_distribution"]:
        cbar_label  = r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]"
        colour_vals = (cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz
        metadata_dates = pd.to_datetime(metadata["date"], format="%Y-%m-%d %H:%M:%S")
        plot_evo_of_freq(colour_vals, metadata_dates, cbar_label, raw_run_dir)

    t0 = time.time()


    # =======================================================================
    # SPECTRUM CUTS
    # =======================================================================

    cut_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'cut_spectra'
    cut_run_dir.mkdir(parents=True, exist_ok=True)

    sset = cut_by_values(sset, cut_min_val = -0.3e6, cut_max_val = 2.3e6)
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    # Plot example trimmed spectra
    plot_spectrum(fper[0]/1e9, specs[0], "Example valid trimmed spectrum", cut_run_dir/"trimmed_spectrum_first.png")
    plot_spectrum(fper[-1]/1e9, specs[-1], "Example valid trimmed spectrum", cut_run_dir/"trimmed_spectrum_last.png")


    # =======================================================================
    # GAIN ALIGNMENT
    # =======================================================================

    gain_alignment = True

    if gain_alignment:
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
    if qc["data_cleaning"]:
        data_clean_dir = run_dir / 'data_cleaning'
        data_clean_dir.mkdir(parents=True, exist_ok=True)

        n_iter = base["n_iterations"]
        counter = 0
        num_clean_graphs = 5
        new_specs = []
        new_freqs = []
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

                if not masked_this_iteration.any() and not out["save_data"]:
                    break
                if counter != num_clean_graphs:
                    counter += 1
                elif counter != 0 and out["save_data"]:
                    continue
                else:
                    break
                if iteration == 1:
                    spec_dir = data_clean_dir / f"spectra_{spec_idx}"
                    spec_dir.mkdir(parents=True, exist_ok=True)
                    

                plot_data_cleaning(freq, spec,metadata, baseline, threshold, 
                                   residuals, spec_idx, masked_this_iteration, 
                                   masked_previously, mask, unmasked, iteration=iteration, base=base, run_dir=run_dir)
            new_specs.append(new_spec)
        specs = new_specs

    # =======================================================================
    # Warm Baseline Removal
    # =======================================================================

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    warm_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /'warm_baseline'
    warm_run_dir.mkdir(parents=True, exist_ok=True)

    spacing_minutes = base["spacing_minutes"]
    dts = metadata["date"]
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
 
    if out["set_average_diagnostics"]:

        # Plot set averaged spectra for all sets on one axis
        colour_vals = np.abs(cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz
        cbar_label  = r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]"
        plot_sets("sets", fper, specs, colour_vals, cbar_label, warm_run_dir, f"Set-averaged spectra — all sets (n = {len(sets)})",
                  "set_averaged_spectra_all.png", plt.cm.viridis, set_sg_fits=None, sets=sets)

        # Plot set average spectra for all sets 3x3
        plot_3x3("mean", sets, set_mean_res, "IF frequency  [MHz]", "PSD  [V²/Hz]", 
                 f"Set-averaged spectra — all sets (n = {len(sets)})", "set_averaged_spectra_all_3x3.png", warm_run_dir)

        # Plot standard deviation of averaged sets againist frequency for all sets 3x3
        plot_3x3("std", sets, set_mean_res, "Standard Deviation  [V²/Hz]", "IF frequency  [MHz]", 
                 f"Standard deviation of averaged spectra against frequency - all sets (n = {len(sets)})",
                 "std_vs_freq_all_3x3.png", warm_run_dir) 

        # Plot standard deviation of averaged sets againist frequency for all sets
        plot_std_freq(sets, set_mean_res, warm_run_dir)
    
        # Plot a histogram of standard deviation of averaged sets for all sets
        n = len(sets)
        plot_hist(data=[np.std([x[0] for x in set], axis=0) for set in sets],
                vline=None, n=n, bins=50, xlabel="Standard Deviation of average  [V²/Hz]", vlabel=None, 
                title=f"Histogram of standard deviation of averaged sets - all sets (n = {len(sets)})",
                cb_label="Mean cavity resonance [GHz]", output_loc=f"{warm_run_dir}/std_hist_all" )


        # Plot average std for each set againist set number
        av_stds = []
        for s, set in enumerate(sets):
            std = np.std([x[0] for x in set], axis=0)
            av_stds.append(np.mean(std))
        plot_std_set_num(av_stds, warm_run_dir)


        for s, set in enumerate(tqdm(sets, desc="Set averaging diagnostic plots")):
        # for g, (freqs, specs) in enumerate(set_avg_spectra):

            # Plot set averaged spectra + the sets spectra per set
            plot_spectra_in_set(set, s, warm_run_dir)

            # Plot set averaged spectra + the sets spectra per set - log plot
            plot_log_spectra_in_set(set, s, warm_run_dir)

            # Plot set averaged spectra with errors per set
            plot_set_average_errors(set, s, warm_run_dir)


            # Plot zoomed set averaged spectra with errors zoomed in per set
            plot_zoom_set_average_errors(set, s, warm_run_dir)
        

            # Plot histogram of each set averaged spectra per set
            mean_val = np.mean([x[0] for x in set])
            med_val = np.median([x[0] for x in set])
            plot_hist(data=np.mean([x[0] for x in set], axis=0), vline=[mean_val, med_val],
                    n=1, bins=100, xlabel="PSD  [V²/Hz]", vlabel=["mean value", "median value"],
                    title=f"Histogram of set averaged set {s}", cb_label=None, output_loc=f"{warm_run_dir}/PSD_histogram_of_set_test{s}")


            # Plot standard deviation of each set average againist frequency per set
            plot_std_against_freq(set, s, set_mean_res, warm_run_dir)

    colour_vals = np.abs(cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz
    cbar_label  = r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]"
    plot_sets("sg_fit", fper, specs, colour_vals, cbar_label, warm_run_dir,
                           "Set-averaged spectra with initial SG fits  (dashed = fit)", "set_averaged_spectra_with_sg_fits.png",
                            cmap=plt.cm.viridis, set_sg_fits=set_sg_fits)


    # -----------------------------------------------------------------------
    # Iterative Sigma Clipping
    # -----------------------------------------------------------------------

    if base["clipping_mode"] == "Claude":
        set_masks = [
            np.zeros(len(avg[0]), dtype=int) if avg is not None else None
            for avg in set_avg_spectra
        ]
    elif base["clipping_mode"] == "Blue":
        set_masks = [
            [np.zeros(len(item[0]), dtype=int) for item in set]
            for set in sets
        ]
    else:
        raise ValueError(f"Clipping mode {base["clipping_mode"]} no found. Did you enter the correct name?")

    sigma_cut = base["sigma_cut"]
    n_iterations = base["n_iterations"]
    for iteration in range(1, n_iterations + 1):
        print(f"\n  --- Iteration {iteration} / {n_iterations} ---")

        if base["clipping_mode"] == "Claude":
            set_masks, set_sg_fits = claude_clipping(
                set_avg_spectra, set_masks, set_sg_fits,
                sigma_cut, base["sg_window_warm"], base["sg_poly_warm"], iteration
                )
            plotting_set_masks = [
                [mask] * len(set) if mask is not None else None
                for mask, set in zip(set_masks, sets)
            ]

        elif base["clipping_mode"] == "Blue":
            set_masks, set_sg_fits = blue_clipping(
                sets, set_masks, set_sg_fits, sigma_cut,
                base["sg_window_warm"], base["sg_poly_warm"], iteration
                )
            plotting_set_masks = set_masks

        plot_iteritive_clipping(set_avg_spectra, plotting_set_masks, set_sg_fits,iteration, warm_run_dir, set_mean_res)


    # --------------------
    # Residuals Plotting
    # --------------------

    if out["clipping_residuals"]:

        clip_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /f'clipping_plots_{base["clipping_mode"]}'
        clip_run_dir.mkdir(parents=True, exist_ok=True)

        for s, fit in enumerate(tqdm(set_sg_fits, desc="Clipping residuals plots")):

            if base["clipping_mode"] == "Claude":

                avg = set_avg_spectra[s]
                if avg is None or fit is None:
                    continue
                freqs, specs = avg
                residuals = specs - fit

                # Plot residuals againist frequency
                plot_claude_residuals(freqs, residuals, s, clip_run_dir)

                # Plot histogram of residuals
                plot_hist(data=residuals[np.isfinite(residuals)], vline=None, n=1, bins=50, xlabel="IF frequency  [MHz]",
                        vlabel=None, title=f"Residuals - set {s} (Claude's clipping method)", cb_label=None, 
                        output_loc=f"{clip_run_dir}/claude_residuals_hist__test{s}.png" )
        


            elif base["clipping_mode"] == "Blue":

                set = sets[s]
                if fit is None or len(set) == 0:
                    continue

                
                # Plot residuals againist frequnecy for each set
                all_residuals = plot_blue_residuals(set, fit, cm.viridis(np.linspace(0, 1, len(set))), s, clip_run_dir)


                # Plot stacked histogram of residuals in each set
                plot_hist(data=[r[np.isfinite(r)] for r in all_residuals],
                        vline=None, n=n, bins=50, xlabel="Residuals  [V²/Hz]", vlabel=None,
                        title=f"Residuals histogram (stacked) — set {s} (Blue's clipping method)", cb_label="Spectrum index in set", 
                        output_loc=f"{clip_run_dir}/blue_residuals_hist_test{s}.png")


    # --------------------
    # Varying Set Size
    # --------------------

    if out["varying_set_size"]:
        specs_set = shifted_spectra
        var_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'varying_set_size'
        var_dir.mkdir(parents=True, exist_ok=True)

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
    specs, fper = finalise_specs(base["clipping_mode"], set_avg_spectra, sets, set_sg_fits)

    plot_sets("baseline_removal", fper, specs, colour_vals, r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]", 
                          warm_run_dir, "Set-averaged spectra with initial SG fits  (dashed = fit)",
                          "spectra_baseline_removed.png", cmap=plt.cm.inferno)

  
    # =======================================================================
    # "Cold" Baseline Removal
    # =======================================================================

    _= remove_baseline(
    spectrum=specs[0],
    window_length=base["sg_window_cold"],
    polyorder=base["sg_poly_cold"],
    subtract_one=True,
    diagnostic={"outfile": run_dir / "baseline_s000_before_after.png",
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
    plot_combination(rf, combined, run_dir)

    # =======================================================================
    # Rebin + Grand Spectrum (SHM template)
    # =======================================================================
    C, K = rb["C"], rb["K"]
    Dr, sr, _ = rebin_ml(combined, sigma_c, C=C)
    freqs_r = rf[:len(Dr)*C:C] + (C//2)*sim["bin_width_hz"]
    f0 = freqs_r[len(freqs_r)//2]
    f0 = np.average(metadata["res_freq"])
    if f0 >= 1e3:
        f0 *= 1e9
    Lq = shm_maxwell_template(K=K, bin_width_hz=C*sim["bin_width_hz"], f0_hz=f0)
    Dg, sg = grand_spectrum_ml(Dr, sr, Lq)
    Dg, sg = Dr, sr

    z = np.zeros_like(Dg); m = np.isfinite(sg) & (sg>0); z[m] = Dg[m]/sg[m]
    plot_grand_spectrum(freqs_r, z, run_dir)

    # =======================================================================
    # Candidates
    # =======================================================================
    ex_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /'candidates_and_exclusion'
    ex_run_dir.mkdir(parents=True, exist_ok=True)

    theta = threshold_for_detection(det["target_snr"], det["confidence"])
    cands, _ = find_candidates(Dg, sg, theta, min_separation=K-1)

    zvals = np.zeros_like(Dg)
    msk = np.isfinite(sg) & (sg > 0)

    
    plot_candidates(freqs_r, zvals, theta, cands, ex_run_dir)


    t1     = time.time()
    total0 = round(t1-t0, 2)
    totals = round(t0-t_sim0, 2)
    if inp["read_input"]:
        print (f"Data Loading Time : {totals} s for {len(metadata['file_name'])} spectra of {len(initial_specs[0])} bins")
    else:
        nbins    = sim["n_bins"]
        nspectra = sim["n_spectra"]
        print (f"Simulation Time : {totals} s for {nspectra} spectra of {nbins} bins")
    print (f"Time from QC to Candidates: {total0} s")

    # =======================================================================
    # Exclusion
    # =======================================================================

    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=det["target_snr"], g0=det["g0"], snr_efficiency=det["snr_eff"])
    plot_exclusion(freqs_r, gmin, outfile=ex_run_dir/"exclusion.png", title="95% CL Exclusion (SHM)")
    with (ex_run_dir/"exclusion.csv").open("w") as fh:
        fh.write("freq_Hz,g_min_rel_to_g0\n")
        for f,g in zip(freqs_r, gmin):
            if np.isfinite(g): fh.write(f"{f},{g}\n")

    print(f"[OK] Run dir: {run_dir}")
    print(f"Candidates flagged: {len(cands)}  (threshold = {theta:.2f}σ)")


if __name__ == "__main__":
    main()