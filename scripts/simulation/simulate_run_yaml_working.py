#!/usr/bin/env python
"""
Simulate a haloscope scan from a YAML config:
- simulation parameters
- optional axion injection
- optional saving of per-spectrum PNGs and spectra.npz
Outputs to: ./output/run_YYYYmmdd_HHMMSS/
"""
from __future__ import annotations
import argparse, datetime, pathlib, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize 
import matplotlib.pyplot as plt
import time
import pandas as pd
import h5py
import matplotlib.dates as mdates
import matplotlib.cm as cm


from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape  import shm_maxwell_template
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit, plot_exclusion
from axion_haloscope.data_quality_working import filter_spectrum_set, too_noisy, power_too_high, metadata_is_zeros, time_filter
from axion_haloscope.io_working import SpectrumSet, SpectrumMetadata, read_hdf5, write_hdf5
from axion_haloscope.width_fq   import width_from_fq


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
            "read_input":       bool(_get(inp, "read_input", False)),
            "directory":        _get(inp, "directory", "scripts/qshs/output/qshs_import"),
            "input_file_name":  _get(inp, "input_file_name", "spectra.h5"),
            "full_data_directory": _get(inp, "full_data_directory", "input/Feb/All"),
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
            "bandwidth_zeros_filter":   bool(_get(qc, "bandwidth_zeros_filter", True)),
            "res_freq_zeros_filter":    bool(_get(qc, "res_freq_zeros_filter", True)),
            "bad_time_filter":          bool(_get(qc, "bad_time_filter", True)),
            "start_time":               _get(qc, "start_time", None),
            "end_time":                 _get(qc, "end_time", None),
        },
        "baseline": {
            "sg_window_warm": int(_get(base, "sg_window_warm", 251)),
            "sg_poly_warm":   int(_get(base, "sg_poly_warm", 2)),
            "sg_window_cold": int(_get(base, "sg_window_cold", 401)),
            "sg_poly_cold":   int(_get(base, "sg_poly_cold", 4)),
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
            "save_data":     bool(_get(out, "save_data", False)),
            "combined_plot": bool(_get(out, "combined_plot", False)),
            "offset_combined_plot": bool(_get(out, "offset_combined_plot", False)),
            "plots_step":    int(_get(out, "plots_step", 1)),   # plot every Nth spectrum
            "max_plots":     out.get("max_plots", None),        # optional int
            "root":          _get(out, "root", "output"),
            "subdir_prefix": _get(out, "subdir_prefix", "run"),
        },
    }
    return cfg

def filter_by_datetime(data, start, end, key="date"):
    '''
    Filters data by a predetermined time range
    '''

    specs, fper, rf, rf_map, metadata = data.spectra, data.freqs_per_spec, data.rf_grid, data.rf_index_map, data.metadata

    dt = np.array([
        datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S") if x is not None else None
        for x in metadata[key]
    ])

    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt   = datetime.datetime.strptime(end,   "%Y-%m-%d %H:%M:%S")

    mask = np.array([(d is not None) and (start_dt <= d <= end_dt) for d in dt])

    print("=" * 60)
    print(f"Timestamp filter: keeping {np.sum(mask)} / {len(mask)} files")
    print("=" * 60)

    spectra  = [b for a, b in zip(mask, specs) if a]
    freqs_per_spec  = [b for a, b in zip(mask, fper) if a]
    rf_index_map  = [b for a, b in zip(mask, rf_map) if a]
    spec_metadata  = {   
        k: [v for keep, v in zip(mask, vals) if keep]
        for k, vals in metadata.items()
    }

    return SpectrumSet(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        rf_grid=rf,
        rf_index_map=rf_index_map,
        metadata=spec_metadata
    ) 

def main():
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    t_sim0 = time.time()
    # Axion injection (center mid-span if not provided)
    ax = None
    if inj["enabled"]:
        total_bins = sim["n_bins"] + (sim["n_spectra"] - 1) * sim["tune_step_bins"]
        f_ax = inj["f_axion_hz"]
        if f_ax is None:
            f_ax = sim["f_start_hz"] + 0.5 * total_bins * sim["bin_width_hz"]
        s_ax = width_from_fq(f_ax)
        ax = AxionParams(f_axion_hz=float(f_ax), sigma_hz=s_ax, total_power=inj["total_power"])
    
    # =======================================================================
    # Data input
    # =======================================================================

    if inp["read_input"]:
        # 1) Read in Data
        directory = inp["directory"]
        input_file_name = inp["input_file_name"]
        sset = read_hdf5(f"{directory}/{input_file_name}")
        specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata
    else:
        # 1) Simulate
        specs, fper, rf, rf_map, metadata = simulate_spectra(
        n_spectra=sim["n_spectra"], n_bins=sim["n_bins"],
        bin_width_hz=sim["bin_width_hz"], f_start_hz=sim["f_start_hz"],
        tune_step_bins=sim["tune_step_bins"], rng_seed=sim["rng_seed"],
        noise_sigma=sim["noise_sigma"], axion=ax
    )

    # =======================================================================
    # Quality Control
    # =======================================================================

    QC_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'QC'
    QC_run_dir.mkdir(parents=True, exist_ok=True)

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
        for s in bad_power:
            invalid_files.append([sset_power.metadata["file_name"][s], "power is too high", sset_power.metadata["date"][s]])
        print(f"[QC]: {len(bad_power)} spectra were removed as power is too high.")
        # Seperate invalid power spectra to plot
        specs_invalid_power, fper_invalid_power, _, _, _ = sset_power.spectra, sset_power.freqs_per_spec, sset_power.rf_grid, sset_power.rf_index_map, sset_power.metadata
        if len(specs_invalid_power) != 0:
            plt.figure(figsize=(9,3))
            plt.plot(fper_invalid_power[0]/1e9, specs_invalid_power[0], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (high power)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_power_raw_spectrum.png", dpi=150); plt.close()

            plt.figure(figsize=(9,3))
            plt.plot(fper_invalid_power[-1]/1e9, specs_invalid_power[-1], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (high power)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_power_raw_spectrum_last.png", dpi=150); plt.close()

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])


    if qc["noise_filter"]:
        sset, sset_noise, kept, bad_noise = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: too_noisy(
                s,
                f,
                md,
                i,
                rms_max=qc["rms_max"],
            ),
        )
        for s in bad_noise:
            invalid_files.append([sset_noise.metadata["file_name"][s], "too noisy", sset_noise.metadata["date"][s]])
        print(f"[QC]: {len(bad_noise)} spectra were removed as too noisy.")
        # Seperate invalid power spectra to plot
        specs_invalid_noise, fper_invalid_noise, _, _, _ = sset_noise.spectra, sset_noise.freqs_per_spec, sset_noise.rf_grid, sset_noise.rf_index_map, sset_noise.metadata
        if len(specs_invalid_noise) != 0:
            plt.figure(figsize=(9,3))
            plt.plot(fper_invalid_noise[0]/1e9, specs_invalid_noise[0], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (too noisey)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_noise_raw_spectrum.png", dpi=150); plt.close()

            plt.figure(figsize=(9,3))
            plt.plot(fper_invalid_noise[-1]/1e9, specs_invalid_noise[-1], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (too noisey)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_noise_raw_spectrum_last.png", dpi=150); plt.close()

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])


    if qc["bad_time_filter"]:
        total_bad_time_filter = 0
        if len(qc["start_time"]) != len(qc["end_time"]):
            print("The lists of start times and end times for cutting data are different lengths, please resolve this issue.")
            sys.exit()
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
                for s in bad_time_filter:
                    invalid_files.append([sset_time_filtered.metadata["file_name"][s], f"known bad data ({qc['start_time'][t]}-{qc['end_time'][t]})" , sset_time_filtered.metadata["date"][s]])
                
                specs_invalid_time, fper_invalid_time, _, _, _ = sset_time_filtered.spectra, sset_time_filtered.freqs_per_spec, sset_time_filtered.rf_grid, sset_time_filtered.rf_index_map, sset_time_filtered.metadata
                if len(specs_invalid_time) != 0:
                    plt.figure(figsize=(9,3))
                    plt.plot(fper_invalid_time[0]/1e9, specs_invalid_time[0], lw=0.6)
                    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
                    plt.title(f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})"); plt.grid(alpha=0.3); plt.tight_layout()
                    plt.savefig(QC_run_dir/f"invalid_time_raw_spectrum_{qc['start_time'][t]}-{qc['end_time'][t]}.png", dpi=150); plt.close()

                    plt.figure(figsize=(9,3))
                    plt.plot(fper_invalid_time[-1]/1e9, specs_invalid_time[-1], lw=0.6)
                    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
                    plt.title(f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})"); plt.grid(alpha=0.3); plt.tight_layout()
                    plt.savefig(QC_run_dir/f"invalid_time_raw_spectrum_last_{qc['start_time'][t]}-{qc['end_time'][t]}.png", dpi=150); plt.close()

                    step = max(1, int(out["plots_step"]))
                    max_plots = None if out["max_plots"] is None else int(out["max_plots"])
            print(f"[QC]: {total_bad_time_filter} spectra were removed as within known bad data times.")


    if qc["bandwidth_zeros_filter"]:
        sset, sset_zeros_bandwidth, kept, bad_zeros_bandwidth = filter_spectrum_set(
            sset,
            predicate=lambda s, f, md, i: metadata_is_zeros(
                s,
                f,
                md,
                i,
                item = "bandwidth",
            ),
        )
        for s in bad_zeros_bandwidth:
            invalid_files.append([sset_zeros_bandwidth.metadata["file_name"][s], "bandwidth data is zeros", sset_zeros_bandwidth.metadata["date"][s]])
        print(f"[QC]: {len(bad_zeros_bandwidth)} spectra were removed as bandwidth were arrays of zeros.")

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
        for s in bad_zeros_res_freq:
            invalid_files.append([sset_zeros_res_freq.metadata["file_name"][s], "res_freq data is zeros", sset_zeros_res_freq.metadata["date"][s]])
        print(f"[QC]: {len(bad_zeros_res_freq)} spectra were removed as res_freq were arrays of zeros.")


    print(f"[QC]: {len(kept)} / {len(kept) + len(invalid_files)} files are valid and suitable for anaylsis, {len(invalid_files)} files are invalid.")
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    # Histogram of data againist time
    valid_files_df = pd.DataFrame(zip(metadata["file_name"] , metadata["date"]))
    valid_files_df[1] = pd.to_datetime(valid_files_df[1], format="%Y-%m-%d %H:%M:%S")
    valid_files_df.to_csv(f"{run_dir}/valid.csv")


    if len(invalid_files) != 0:
        invalid_files_df = pd.DataFrame(invalid_files)
        invalid_files_df[2] = pd.to_datetime(invalid_files_df[2], format="%Y-%m-%d %H:%M:%S")
        invalid_metadata = invalid_files_df[invalid_files_df[1] == "metadata is missing"]
        invalid_high_power = invalid_files_df[invalid_files_df[1] == "power is too high"]
        invalid_high_noise = invalid_files_df[invalid_files_df[1] == "too noisy"]
        invalid_power_zeros = invalid_files_df[invalid_files_df[1] == "power spectra is zeros"]
        invalid_bandwidth_zeros = invalid_files_df[invalid_files_df[1] == "bandwidth data is zeros"]
        invalid_res_freq_zeros = invalid_files_df[invalid_files_df[1] == "res_freq data is zeros"]

        start_date = min( valid_files_df[1].min(), invalid_files_df[2].min())
        end_date = max( valid_files_df[1].max(), invalid_files_df[2].max())
        time_interval = pd.Timedelta(hours=24)
        bin_num = int( (end_date - start_date) / time_interval)

        plt.figure(figsize=(18,6))
        plt.hist((valid_files_df[1], invalid_metadata[2], invalid_high_power[2], invalid_high_noise[2], invalid_power_zeros[2], invalid_bandwidth_zeros[2], invalid_res_freq_zeros[2])
                , bin_num, range=(start_date, end_date), stacked=True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.legend(["valid files", "missing metadata", "power too high", "too noisy", "power is zeros", "bandwidth is zeros", "res freq is zeros"])
        plt.xlabel("Date")
        plt.ylabel("Number of files")
        plt.title("Spectra files per day (before timestamp filter)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir/"spectra_hist.png", dpi=150)
        plt.close()

    else:
        start_date = valid_files_df[1].min()
        end_date = valid_files_df[1].max()
        time_interval = pd.Timedelta(hours=0.1)
        bin_num = int( (end_date - start_date) / time_interval)

        plt.figure(figsize=(18,6))
        plt.hist(valid_files_df[1], bin_num, range=(start_date, end_date), stacked=True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.legend("valid files")
        plt.xlabel("Date")
        plt.ylabel("Number of files")
        plt.title("Spectra files per day (before timestamp filter)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir/"spectra_hist.png", dpi=150)
        plt.close()

    # Plot of total number of events againist time
    fig, ax = plt.subplots(figsize = (13,5))
    if len(invalid_files) != 0:
        invalid_files_df = invalid_files_df.sort_values(by=[2])
        count_invalid = range(1, len(invalid_files_df[2]) + 1)
        all_files_df = pd.concat([valid_files_df[1], invalid_files_df[2]])
        all_files_df = all_files_df.sort_values()
        ax.plot(invalid_files_df[2], count_invalid, label='invalid files', color="red")

    else:
        all_files_df = valid_files_df[1]
        all_files_df = all_files_df.sort_values()
    
    valid_files_df = valid_files_df.sort_values(by=[1])
    count_valid = range(1, len(valid_files_df[1]) + 1)
    count_all = range(1,len(all_files_df) + 1)

    ax.plot(valid_files_df[1], count_valid, label="valid files", color="green")
    ax.plot(all_files_df, count_all, label='all files', linestyle='dashed', color="orange")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.set_xlabel("Date-Time")
    ax.set_ylabel("Events")
    ax.set_title(f"Evolution of number of events w.r.t. Time")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/events_agaisnt_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Export the file spectrum set used for anaylsis to h5 file
    out_h5 = f"{run_dir}/final_converted_spectra.h5"
    write_hdf5(sset, out_h5)
    print(f"[QSHS] Final SpectrumSet saved to: {out_h5}")

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

    sset = filter_by_datetime(
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


    # ===================
    # Spectra Plotting
    # ===================

    raw_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}'/ 'raw_spectra'
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    # Calculate difference in resonant frequency of the cavity between the spectra
    res_freq_diff = []
    for f in res_freqs:
        difference = f - res_freqs[0]
        res_freq_diff.append(difference)

    # Always save one valid example raw spectrum
    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(raw_run_dir/"valid_raw_spectrum.png", dpi=150); plt.close()

    plt.figure(figsize=(9,3))
    plt.plot(fper[-1]/1e9, specs[-1], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(raw_run_dir/"valid_raw_spectrum_last.png", dpi=150); plt.close()

    step = max(1, int(out["plots_step"]))
    max_plots = None if out["max_plots"] is None else int(out["max_plots"])

    # Optional: save per-spectrum PNGs + spectra.npz for valid data
    if out["save_data"]:
        count = 0
        for i, (freqs, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break
            fig, axp = plt.subplots(figsize=(9,3))
            axp.plot(freqs/1e6, spec, lw=0.6)
            axp.set(xlabel="Frequency [MHz]", ylabel="Raw Power [arb]", title=f"Spectrum {i:03d}")
            axp.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(raw_run_dir / f"spectrum_{i:03d}.png", dpi=120)
            plt.close(fig)
            count += 1
        np.savez(run_dir/"spectra.npz", spectra=np.array(specs), freqs=fper, rf_grid=rf)

    # Optional: plot all valid/invalid raw spectra in one figure
    if out["combined_plot"]:
        plt.figure(figsize=(9,3))
        for i, (freqs, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break
            plt.plot(freqs/1e9, spec, lw=0.6)
        plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
        plt.title("All valid raw spectra"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(raw_run_dir/"raw_valid_spectrum_all.png", dpi=150); plt.close()

        if len(specs_invalid_power) != 0:
            plt.figure(figsize=(9,3))
            for i, (freqs, spec) in enumerate(zip(fper_invalid_power, specs_invalid_power)):
                if i % step != 0:
                    continue
                if max_plots is not None and count >= max_plots:
                    break
                plt.plot(freqs/1e9, spec, lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("All invalid raw spectra"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(raw_run_dir/"raw_invalid_spectrum_all.png", dpi=150); plt.close()

    # Optional: plot all valid raw spectra in one figure with offset
    if out["offset_combined_plot"]:
        # Plot the resonance frequency offset againist the spectrum index
        plt.figure(figsize=(9,3))
        plt.scatter( range(len(res_freq_diff)),res_freq_diff)
        plt.xlabel("Spectrum Index"); plt.ylabel("Resonance Frequency Offset [Hz]")
        plt.title("Resonance Frequency Offset vs Spectrum Index"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(raw_run_dir/"res_freq_offset_vs_index.png", dpi=150); plt.close()

        # Combine the offset spectra into one figure
        plt.figure(figsize=(9,3))
        for i, (freqs, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break
            plt.plot((freqs/1e9 + res_freq_diff[i]), spec, lw=0.6)
        plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
        plt.title("All valid spectra offset"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(raw_run_dir/"raw_spectrum_all_valid_offset.png", dpi=150); plt.close()  

    # Optional: plot all valid raw spectra in one figure with offset

    colour_vals = np.abs(cw_freqs - res_freqs) / 1e9  # GHz → Hz


    # DEBUGGING - show the total number of injections
    # print(np.unique(np.round(colour_vals, 10), return_counts=True))
    #if out["injection_histogram"]:
    # Plot the resonance frequency offset againist the spectrum index
    plt.figure(figsize=(9,3))
    plt.hist(colour_vals, bins = len(np.unique(np.round(colour_vals, 10))))
    plt.xlabel("Shifted Injected Frequency"); plt.ylabel("Count")
    plt.title("Injected Frequency Histogram"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(raw_run_dir/"shift.png", dpi=150); plt.close()

    # Combine the offset spectra into one figure
    plt.figure(figsize=(9,3))
    for i, (freqs, spec) in enumerate(zip(fper, specs)):
        if i % step != 0:
            continue
        if max_plots is not None and count >= max_plots:
            break
        plt.plot((freqs/1e9 + res_freq_diff[i]), spec, lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("All valid spectra offset"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(raw_run_dir/"raw_spectrum_all_valid_offset.png", dpi=150); plt.close() 

 
    t0 = time.time()

    # =======================================================================
    # SPECTRUM CUTS
    # =======================================================================

    cut_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'trimming'
    cut_run_dir.mkdir(parents=True, exist_ok=True)

    cut_min_val = -0.3e6
    cut_max_val = 2.3e6
    cut_min_idx = np.abs(fper[0] - cut_min_val).argmin()
    cut_max_idx = np.abs(fper[0] - cut_max_val).argmin()

    new_specs = []
    new_freqs = []
    new_rf_map = []
    for spec, freq, rf_vals in zip(specs, fper, rf_map):
        x = np.where(freq == 0)[0]
        for j in x:
            for i in range(2, -1, -1):
                spec[j+i] = spec[j+i+1]
                spec[j-i-1] = spec[j-i-2]


        spec = spec[cut_min_idx:cut_max_idx]
        freq = freq[cut_min_idx:cut_max_idx]
        rf_vals = rf_vals[cut_min_idx:cut_max_idx]
        

        new_specs.append(spec)
        new_freqs.append(freq)
        new_rf_map.append(rf_vals)

    specs = new_specs
    fper = new_freqs
    rf = rf[cut_min_idx:cut_max_idx]
    rf_map = new_rf_map

    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(cut_run_dir/"trimmed_spectrum_first.png", dpi=150); plt.close()

    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(cut_run_dir/"trimmed_spectrum_last.png", dpi=150); plt.close()


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
    # Warm Baseline Removal
    # =======================================================================

    warm_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /'warm_baseline'
    warm_run_dir.mkdir(parents=True, exist_ok=True)

    spacing_minutes = 30
    date_times = metadata["date"]
    dts=[]
    for date_time in date_times:
        try:
            dt = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
            dts.append(dt)
        except ValueError as e:
            print(f"{date_time} -> {e}")

    groups = []
    n = len(dts)
    threshold = spacing_minutes * 60  # seconds
    i = 0
    while i < n:
        j = i + 1
        while j < n and (dts[j] - dts[i]).total_seconds() < threshold:
            j += 1
        groups.append([[specs[k], fper[k], metadata["res_freq"][k]] for k in range(i, j)])
        i = j

    

    def interpolate_nans(y):
        y = np.asarray(y, dtype=float)
        nans = np.isnan(y)
        if nans.any():
            if nans.all():
                return np.nan_to_num(y)  
            x = np.arange(len(y))
            y = y.copy()
            y[nans] = np.interp(x[nans], x[~nans], y[~nans])
        return y
    

    _cmap_g_1 = plt.cm.viridis
    _cmap_g_2 = plt.cm.inferno



    _group_mean_res = np.array([
    np.nanmean([item[2] for item in group], axis=0)
    for group in groups
    ])
    
    
    _finite_res = _group_mean_res[np.isfinite(_group_mean_res)]
    _norm_res = Normalize(
        vmin=np.nanmin(_finite_res) if len(_finite_res) else 0,
        vmax=np.nanmax(_finite_res) if len(_finite_res) else 1,
    )
    def _gcol_1(g):
        v = _group_mean_res[g]
        if not np.isfinite(v):
            return "grey"
        return _cmap_g_1(_norm_res(v))
    
    def _gcol_2(g):
        v = _group_mean_res[g]
        if not np.isfinite(v):
            return "grey"
        return _cmap_g_2(_norm_res(v))


    fig, ax = plt.subplots(figsize=(13, 5))
    for g, group in enumerate(groups):
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), alpha=0.8, color=_gcol_1(g), label =f"Grp {g}")
    sm_res = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Set-averaged spectra — all groups")
    plt.tight_layout()
    plt.savefig(f"{warm_run_dir}/set_averaged_spectra.png", dpi = 150, bbox_inches='tight')
    plt.close()


    for g, group in enumerate(groups):
        # Plot set averaged + the set
        fig, ax = plt.subplots(figsize=(13, 5))
        greys = cm.Greys(np.linspace(0.3, 0.9, len(group)))
        for i, x in enumerate(group):
            ax.plot(x[1]/1e6, x[0], color=greys[i])
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), alpha=0.8, color="red")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Set-averaged spectra and set spectra — group {g}")
        plt.tight_layout()
        plt.savefig(f"{warm_run_dir}/set_and_average_spectra_{g}.png", dpi = 150, bbox_inches='tight')
        plt.close()

        # Plot set averaged spectra with errors
        fig, ax = plt.subplots(figsize=(26, 10))
        ax.errorbar(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), np.std([x[0] for x in group], axis=0), alpha=0.8, ecolor="blue", color="red")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Set-averaged spectra with errors — group {g}")
        plt.tight_layout()
        plt.savefig(f"{warm_run_dir}/set_averaged_spectra_errors_{g}.png", dpi = 150, bbox_inches='tight')
        plt.close()

        # Plot zoomed set averaged spectra with errors zoomed in
        fig, ax = plt.subplots(figsize=(39, 15))
        ax.errorbar(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), np.std([x[0] for x in group], axis=0), alpha=0.8, ecolor="blue", color="red")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Set-averaged spectra with errors — group {g} (zoomed)")
        plt.tight_layout()
        plt.xlim(1.5, 2)
        plt.ylim(1.78e-10, 1.84e-10)
        plt.savefig(f"{warm_run_dir}/set_averaged_spectra_errors_{g}_zoom.png", dpi = 150, bbox_inches='tight')
        plt.close()

        sys.exit()


    group_sg_fits = []
    for g, group in enumerate(groups):
        if group is None:
            group_sg_fits.append(None)
            continue

        _, baseline = remove_baseline(
                spectrum=np.mean([x[0] for x in group], axis=0),
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
                )
        group_sg_fits.append(baseline)


    fig, ax = plt.subplots(figsize=(13, 5))
    for g, (group, fit) in enumerate(zip(groups, group_sg_fits)):
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0),   lw=1.0, alpha=0.55, color=_gcol_1(g), label=f"Grp {g}")
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, fit, lw=1.8, alpha=0.95, color=_gcol_1(g), linestyle="--")
    sm_res2 = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
    sm_res2.set_array([])
    fig.colorbar(sm_res2, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Group-averaged spectra with initial SG fits  (dashed = fit)")
    plt.tight_layout()
    plt.savefig(f"{warm_run_dir}/group_averaged_spectra_with_sg_fits.png", dpi = 150, bbox_inches='tight')
    plt.close()



    sigma_cut = 3.5
    specs = []
    fper = []
    masked_total = [[] for _ in groups]

    persistent_masks = [
        [np.zeros(len(item[0]), dtype=bool) for item in group]
        for group in groups
    ]

    for run in range(1, 4):
        masked_by_group = []
        new_groups = []
        new_group_sg_fits = []
        new_persistent_masks = []

        for group_idx, group in enumerate(groups):
            group_masks = persistent_masks[group_idx]   

            spectra_stack = np.array([x[0] for x in group])
            mask_stack    = np.array(group_masks)

            masked_stack = np.ma.masked_array(spectra_stack, mask=mask_stack)
            average_spectra = masked_stack.mean(axis=0).filled(np.nan)
            sd_spectra      = masked_stack.std(axis=0).filled(np.nan)

            average_for_fit = interpolate_nans(average_spectra)
            _, baseline = remove_baseline(
                spectrum=average_for_fit,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
            )
            new_group_sg_fits.append(baseline)

            masked_new = []
            new_group = []
            new_group_masks = []

            for spec_idx, (spectra, frequencies, res_freq) in enumerate(group):
                prev_mask = group_masks[spec_idx]

                deviation = np.abs(spectra - baseline)
                new_flags = (deviation > sigma_cut * sd_spectra) & ~prev_mask

                cum_mask = prev_mask | new_flags

                spec_m = np.ma.masked_array(spectra, cum_mask)
                freq_m = np.ma.masked_array(frequencies, cum_mask)

                cleaned_spec = interpolate_nans(spec_m.filled(np.nan))
                cleaned_freq = interpolate_nans(freq_m.filled(np.nan))

                newly_masked_idx = np.where(new_flags)[0]
                for idx in newly_masked_idx:
                    masked_new.append([frequencies[idx], spectra[idx]])

                new_group.append([cleaned_spec, cleaned_freq, res_freq])
                new_group_masks.append(cum_mask)

            masked_by_group.append(masked_new)
            new_groups.append(new_group)
            new_persistent_masks.append(new_group_masks)

        fig, ax = plt.subplots(figsize=(13, 5))
        for g, (group, fit) in enumerate(zip(groups, new_group_sg_fits)):
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), lw=1.0, alpha=0.55, color=_gcol_1(g), label=f"Grp {g}")
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, fit, lw=1.8, alpha=0.95, color=_gcol_1(g), linestyle="--")
            pts = np.array(masked_by_group[g])
            if pts.size:
                ax.scatter(pts[:, 0]/1e6, pts[:, 1], marker = ".", color=_gcol_2(g), zorder=5)
            old_pts = np.array(masked_total[g])
            if old_pts.size:
                ax.scatter(old_pts[:, 0]/1e6, old_pts[:, 1], c="grey", zorder=4)

        sm_res1 = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
        sm_res1.set_array([])
        fig.colorbar(sm_res1, ax=ax, label="Mean cavity resonance  [GHz]", pad=0.02)

        sm_res2 = ScalarMappable(cmap=_cmap_g_2, norm=_norm_res)
        sm_res2.set_array([])
        fig.colorbar(sm_res2, ax=ax, label="Mean cavity resonance  [GHz]", pad=0.10)

        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title("Group-averaged spectra with initial SG fits  (dashed = fit)")
        plt.tight_layout()
        plt.savefig(f"{warm_run_dir}/masked_bin_iteration_{run}.png", dpi=150, bbox_inches='tight')
        plt.close()

        for g in range(len(groups)):
            masked_total[g].extend(masked_by_group[g])
        groups = new_groups
        group_sg_fits = new_group_sg_fits
        persistent_masks = new_persistent_masks

    specs = []
    fper = []
    for group, baseline in zip(groups, group_sg_fits):
        group_spectra = np.array([item[0] for item in group])
        group_freqs = np.array([item[1] for item in group])

        specs.extend(group_spectra / baseline)
        fper.extend(group_freqs)

    colour_vals = np.abs(cw_freqs - res_freqs) / 1e9  # GHz → Hz


    # DEBUGGING - show the total number of injections
    # print(np.unique(np.round(colour_vals, 10), return_counts=True))


    cbar_label  = r"$|f_{\rm CW} - f_{\rm res}|$  [Hz]"
    cmap        = plt.cm.inferno
    _finite     = colour_vals[np.isfinite(colour_vals)]
    norm = Normalize(
        vmin=np.percentile(_finite,  0),
        vmax=np.percentile(_finite, 100),
    )

    fig, ax = plt.subplots(figsize = (13,5))
    for spec, freq, cv in zip(specs, fper, colour_vals):
        #ax.scatter(freqs, specs, color=_gcol_2(group_idx))
        ax.plot(freq, spec, linestyle="", marker="o", markersize=3, color=cmap(norm(cv)), alpha=0.7)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label)

    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Group-averaged spectra with initial SG fits  (dashed = fit)")
    plt.tight_layout()
    plt.savefig(f"{warm_run_dir}/spectra-baseline_removed.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize = (13,5))
    ax.plot(metadata["date"], colour_vals)
    ax.set_xlabel("Date-Time")
    ax.set_ylabel(cbar_label)
    ax.set_title(f"Evolution of {cbar_label} w.r.t. Time")
    plt.tight_layout()
    plt.savefig(f"{warm_run_dir}/evolution_of_frequency.png", dpi=150, bbox_inches='tight')
    plt.close()

    sys.exit()







    # =======================================================================
    # Cold Baseline Removal
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

    # 3) combine
    combined, sigma_c, counts = combine_ml(proc, rf_map_new, total_rf_bins=len(rf))
    plt.figure(figsize=(10,3))
    plt.plot(rf/1e9, combined, lw=0.8, color="black", label="combined")
    plt.title("Combined spectrum (baseline-removed)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Excess power [arb]"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run_dir/"combined.png", dpi=150); plt.close()

    # 4) rebin + grand spectrum (SHM template)
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
    plt.figure(figsize=(10,3))
    plt.plot(freqs_r/1e9, z, lw=0.8)
    plt.title("Grand spectrum z-score (SHM matched filter)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("z"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run_dir/"grand_z.png", dpi=150); plt.close()

    # 5) candidates
    theta = threshold_for_detection(det["target_snr"], det["confidence"])
    cands, _ = find_candidates(Dg, sg, theta, min_separation=K-1)
    # After: cands, z = find_candidates(Dg, sg, theta, min_separation=K-1)

    fig, ax = plt.subplots(figsize=(10, 3))

    # plot the z-score trace
    zvals = np.zeros_like(Dg)
    msk = np.isfinite(sg) & (sg > 0)
    zvals[msk] = Dg[msk] / sg[msk]
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


    t1     = time.time()
    total0 = round(t1-t0, 2)
    totals = round(t0-t_sim0, 2)
    nbins    = sim["n_bins"]
    nspectra = sim["n_spectra"]
    print (f"Simulation Time : {totals} s for {nspectra} spectra of {nbins} bins")
    print (f"Time from QC to Candidates: {total0} s")

    # 6) exclusion
    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=det["target_snr"], g0=det["g0"], snr_efficiency=det["snr_eff"])
    plot_exclusion(freqs_r, gmin, outfile=run_dir/"exclusion.png", title="95% CL Exclusion (SHM)")
    with (run_dir/"exclusion.csv").open("w") as fh:
        fh.write("freq_Hz,g_min_rel_to_g0\n")
        for f,g in zip(freqs_r, gmin):
            if np.isfinite(g): fh.write(f"{f},{g}\n")

    print(f"[OK] Run dir: {run_dir}")
    print(f"Candidates flagged: {len(cands)}  (threshold = {theta:.2f}σ)")


if __name__ == "__main__":
    main()