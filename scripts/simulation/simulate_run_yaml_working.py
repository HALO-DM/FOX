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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize 
import matplotlib.pyplot as plt
import time
import pandas as pd
import shutil
import h5py
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm


from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape  import shm_maxwell_template
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit, plot_exclusion
from axion_haloscope.data_quality_working import filter_spectrum_set, too_noisy, power_too_high, metadata_is_zeros, time_filter, small_bandwidth
from axion_haloscope.io_working import SpectrumSet, SpectrumMetadata, read_hdf5, write_hdf5
from axion_haloscope.sigma_clipping import claude_clipping, blue_clipping, finalise_specs
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
            "small_bandwidth_filter":   bool(_get(qc, "small_bandwidth_filter", True)),
            "bw_min":                   float(_get(qc, "bw_min", 0.00027)),
            "bandwidth_zeros_filter":   bool(_get(qc, "bandwidth_zeros_filter", True)),
            "res_freq_zeros_filter":    bool(_get(qc, "res_freq_zeros_filter", True)),
            "cw_freq_zeros_filter":     bool(_get(qc, "cw_freq_zeros_filter", True)),
            "bad_time_filter":          bool(_get(qc, "bad_time_filter", True)),
            "start_time":               _get(qc, "start_time", None),
            "end_time":                 _get(qc, "end_time", None),
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
            "save_data":     bool(_get(out, "save_data", False)),
            "combined_plot": bool(_get(out, "combined_plot", False)),
            "offset_combined_plot": bool(_get(out, "offset_combined_plot", False)),
            "injection_distribution": bool(_get(out, "injection_distribution", False)),
            "set_average_diagnostics": bool(_get(out, "set_average_diagnostics", False)),
            "clipping_residuals": bool(_get(out, "clipping_residuals", False)),
            "varying_group_size": bool(_get(out, "varying_group_size", False)),
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

    removed = [[metadata["file_name"][i], "not in good time range", metadata[key][i]]
        for i, keep in enumerate(mask) if not keep]

    invalid = list(metadata.get("invalid_files", []))
    invalid_all = invalid + removed

    spec_metadata = {
        k: (invalid_all if k == "invalid_files" else [val for keep, val in zip(mask, v) if keep])
        for k, v in metadata.items()
    }

    return SpectrumSet(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        rf_grid=rf,
        rf_index_map=rf_index_map,
        metadata=spec_metadata
    ) 

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
        # Axion injection (center mid-span if not provided)
        ax = None
        if inj["enabled"]:
            total_bins = sim["n_bins"] + (sim["n_spectra"] - 1) * sim["tune_step_bins"]
            f_ax = inj["f_axion_hz"]
            if f_ax is None:
                f_ax = sim["f_start_hz"] + 0.5 * total_bins * sim["bin_width_hz"]
            s_ax = width_from_fq(f_ax)
            ax = AxionParams(f_axion_hz=float(f_ax), sigma_hz=s_ax, total_power=inj["total_power"])

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

    print("=" * 60)
    print(f"Quality Control")
    print("=" * 60)

    QC_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'quality_control'
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
            plt.figure(figsize=(13, 7))
            plt.plot(fper_invalid_power[0]/1e9, specs_invalid_power[0], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (high power)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_power_raw_spectrum.png", dpi=150); plt.close()

            plt.figure(figsize=(13, 7))
            plt.plot(fper_invalid_power[-1]/1e9, specs_invalid_power[-1], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (high power)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_power_raw_spectrum_last.png", dpi=150); plt.close()

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
        # Seperate invalid power spectra to plot
        specs_invalid_noise, fper_invalid_noise = sset_noise.spectra, sset_noise.freqs_per_spec
        if len(specs_invalid_noise) != 0:
            plt.figure(figsize=(13, 7))
            plt.plot(fper_invalid_noise[0]/1e9, specs_invalid_noise[0], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (too noisey)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_noise_raw_spectrum.png", dpi=150); plt.close()

            plt.figure(figsize=(13, 7))
            plt.plot(fper_invalid_noise[-1]/1e9, specs_invalid_noise[-1], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (too noisey)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_noise_raw_spectrum_last.png", dpi=150); plt.close()

            step = max(1, int(out["plots_step"]))
            max_plots = None if out["max_plots"] is None else int(out["max_plots"])


    # QC: Cut spectra that are within known bad times
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
                for idx, s in enumerate(bad_time_filter):
                    invalid_files.append([sset_time_filtered.metadata["file_name"][idx], f"known bad data ({qc['start_time'][t]}-{qc['end_time'][t]})" , sset_time_filtered.metadata["date"][idx]])
                
                specs_invalid_time, fper_invalid_time = sset_time_filtered.spectra, sset_time_filtered.freqs_per_spec
                if len(specs_invalid_time) != 0:
                    plt.figure(figsize=(13, 7))
                    plt.plot(fper_invalid_time[0]/1e9, specs_invalid_time[0], lw=0.6)
                    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
                    plt.title(f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})"); plt.grid(alpha=0.3); plt.tight_layout()
                    plt.savefig(QC_run_dir/f"invalid_time_raw_spectrum_{qc['start_time'][t]}-{qc['end_time'][t]}.png", dpi=150); plt.close()

                    plt.figure(figsize=(13, 7))
                    plt.plot(fper_invalid_time[-1]/1e9, specs_invalid_time[-1], lw=0.6)
                    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
                    plt.title(f"Example invalid raw spectrum (invalid time {qc['start_time'][t]}-{qc['end_time'][t]})"); plt.grid(alpha=0.3); plt.tight_layout()
                    plt.savefig(QC_run_dir/f"invalid_time_raw_spectrum_last_{qc['start_time'][t]}-{qc['end_time'][t]}.png", dpi=150); plt.close()

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
            plt.figure(figsize=(13, 7))
            plt.plot(fper_invalid_bandwidth[0]/1e9, specs_invalid_bandwidth[0], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (small bandwidth)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_bandwidth_raw_spectrum.png", dpi=150); plt.close()

            plt.figure(figsize=(13, 7))
            plt.plot(fper_invalid_bandwidth[-1]/1e9, specs_invalid_bandwidth[-1], lw=0.6)
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
            plt.title("Example invalid raw spectrum (small bandwidth)"); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(QC_run_dir/"invalid_bandwidth_raw_spectrum_last.png", dpi=150); plt.close()

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

            plt.figure(figsize=(13, 7))
            plt.scatter(bad_dates_sorted, bad_bandwidths_sorted, color="firebrick", label="removed (below min threshold)")
            plt.scatter(good_dates[good_order], good_bandwidths[good_order], color="steelblue", alpha=0.6, label="kept (above min threshold)")
            plt.axhline(qc["bw_min"], color="black", linestyle="--", linewidth=1, label=f"bw_min = {qc['bw_min']:.4g}")
            plt.xlabel("Date")
            plt.ylabel("Bandwidth [Hz]")
            plt.title("Removed spectra: bandwidth below threshold")
            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.ylim(0, 0.006)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(QC_run_dir / "bad_bandwidth_vs_time.png", dpi=150)
            plt.close()


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

        plt.figure(figsize=(18,6))
        plt.hist((valid_files_df[1], invalid_metadata[2], invalid_high_power[2], invalid_high_noise[2], invalid_power_zeros[2], invalid_bandwidth[2], invalid_res_freq_zeros[2], invalid_cw_freq_zeros[2])
                , bin_num, range=(start_date, end_date), stacked=True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.legend(["valid files", "missing metadata", "power too high", "too noisy", "power is zeros", "bandwidth is too small", "res freq is zeros", "cw freq is zeros"])
        plt.xlabel("Date")
        plt.ylabel("Number of files")
        plt.title("Spectra files per day (before timestamp filter)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(raw_run_dir/"spectra_hist.png", dpi=150)
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
        plt.savefig(raw_run_dir/"spectra_hist.png", dpi=150)
        plt.close()


    # Plot of total number of events againist time
    fig, ax = plt.subplots(figsize=(13, 7))
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

        ax.plot(invalid_dates, count_invalid, label='invalid files', color="red")

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
    plt.savefig(f"{raw_run_dir}/events_agaisnt_time.png", dpi=150, bbox_inches='tight')
    plt.close()


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
    plt.savefig(QC_run_dir/"rms_againist_time_data.png", dpi=150); plt.close()

    
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

        plt.figure(figsize=(18,6))
        plt.hist((valid_files_df[1], invalid_metadata[2], invalid_high_power[2], invalid_high_noise[2], invalid_power_zeros[2], invalid_bandwidth[2], invalid_res_freq_zeros[2], invalid_time[2])
                , bin_num, range=(start_date, end_date), stacked=True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.legend(["valid files", "missing metadata", "power too high", "too noisy", "power is zeros", "bandwidth is too small", "res freq is zeros", "invalid time filter"])
        plt.xlabel("Date")
        plt.ylabel("Number of files")
        plt.title("Spectra files per day (post timestamp filter)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(raw_run_dir/"spectra_hist_time_cut.png", dpi=150)
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
        plt.title("Spectra files per day (post timestamp filter)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(raw_run_dir/"spectra_hist_time_cut.png", dpi=150)
        plt.close()


    # Plot of total number of events againist time (post time filter)
    fig, ax = plt.subplots(figsize = (13,5))
    if len(invalid_files) != 0:
        invalid_files_df = invalid_files_df.sort_values(by=[2])
        count_invalid = range(1, len(invalid_files_df[2]) + 1)
        all_files_df = pd.concat([valid_files_df[1], invalid_files_df[2]])
        all_files_df = all_files_df.sort_values()
        overall_end = max(valid_files_df[1].max(), invalid_files_df[2].max())

        invalid_dates = list(invalid_files_df[2])
        if invalid_dates[-1] < overall_end:
            invalid_dates.append(overall_end)
            count_invalid.append(count_invalid[-1])
        ax.plot(invalid_dates, count_invalid, label='invalid files', color="red")

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

    
    ax.plot(valid_dates, count_valid, label="valid files", color="green")
    ax.plot(all_dates, count_all, label='all files', linestyle='dotted', color="orange")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.set_xlabel("Date-Time")
    ax.set_ylabel("Events")
    ax.set_title(f"Evolution of number of events w.r.t. time (post time filter)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{raw_run_dir}/events_agaisnt_time_time_cut.png", dpi=150, bbox_inches='tight')
    plt.close()

    
    # =============================================================
    # Spectra Plotting
    # =============================================================

    # Calculate difference in resonant frequency of the cavity between the spectra
    res_freq_diff = []
    for f in res_freqs:
        difference = f - res_freqs[0]
        res_freq_diff.append(difference)


    # Always save one valid example raw spectrum
    plt.figure(figsize=(13, 7))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(raw_run_dir/"valid_raw_spectrum.png", dpi=150); plt.close()

    plt.figure(figsize=(13, 7))
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
            fig, axp = plt.subplots(figsize=(13, 7))
            axp.plot(freqs/1e6, spec, lw=0.6)
            axp.set(xlabel="Frequency [MHz]", ylabel="Raw Power [arb]", title=f"Spectrum {i:03d}")
            axp.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(raw_run_dir / f"spectrum_{i:03d}.png", dpi=120)
            plt.close(fig)
            count += 1
        np.savez(run_dir/"spectra.npz", spectra=np.array(specs), freqs=fper, rf_grid=rf)


    # Optional: plot all valid/invalid raw spectra in one figure
    if out["combined_plot"]:
        plt.figure(figsize=(13, 7))
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
            plt.figure(figsize=(13, 7))
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
        plt.figure(figsize=(13, 7))
        plt.scatter( range(len(res_freq_diff)),res_freq_diff)
        plt.xlabel("Spectrum Index"); plt.ylabel("Resonance Frequency Offset [Hz]")
        plt.title("Resonance Frequency Offset vs Spectrum Index"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(raw_run_dir/"res_freq_offset_vs_index.png", dpi=150); plt.close()

        # Combine the offset spectra into one figure
        plt.figure(figsize=(13, 7))
        for i, (freqs, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break
            plt.plot((freqs/1e9 + res_freq_diff[i]), spec, lw=0.6)
        plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
        plt.title("All valid spectra offset"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(raw_run_dir/"raw_spectrum_all_valid_offset.png", dpi=150); plt.close()  


    # Plot injected frequency distrubtion (frequency againist time)
    if out["injection_distribution"]:
        cbar_label  = r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]"
        fig, ax = plt.subplots(figsize = (13,5))
        colour_vals = (cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz
        metadata_dates = pd.to_datetime(metadata["date"], format="%Y-%m-%d %H:%M:%S")
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

    t0 = time.time()


    # =======================================================================
    # SPECTRUM CUTS
    # =======================================================================

    cut_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'cut_spectra'
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

    # Plot example trimmed spectra
    plt.figure(figsize=(13, 7))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(cut_run_dir/"trimmed_spectrum_first.png", dpi=150); plt.close()

    plt.figure(figsize=(13, 7))
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

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    warm_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /'warm_baseline'
    warm_run_dir.mkdir(parents=True, exist_ok=True)

    spacing_minutes = base["spacing_minutes"]
    date_times = metadata["date"]
    dts=[]
    for date_time in date_times:
        try:
            dt = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
            dts.append(dt)
        except ValueError as e:
            print(f"{date_time} -> {e}")

    # -----------------------------------------------------------------------
    # Grouping
    # -----------------------------------------------------------------------

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


    # -----------------------------------------------------------------------
    # Group Averaging
    # -----------------------------------------------------------------------

    group_avg_spectra = []
    for g, group in enumerate(groups):
        if group is None:
            group_avg_spectra.append(None)
            continue
        
        group_avg_spectra.append((np.mean([x[1] for x in group], axis=0), np.mean([x[0] for x in group], axis=0)))

    
    # -----------------------------------------------------------------------
    # Group Average Baseline Fitting
    # -----------------------------------------------------------------------

    group_sg_fits = []
    for _, spec_avg in group_avg_spectra:
        if not spec_avg.any():
            group_sg_fits.append(None)
            continue

        _, baseline = remove_baseline(
                spectrum=spec_avg,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
                )
        group_sg_fits.append(baseline)

    # -----------------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------------

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


    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

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

    if out["set_average_diagnostics"]:

        # Plot set averaged spectra for all sets on one axis
        fig, ax = plt.subplots(figsize=(13, 7))
        # for g, (freqs, specs) in enumerate(group_avg_spectra):
            # ax.plot(freqs/1e6, specs, alpha=0.8, color=_gcol_1(g), label =f"Grp {g}")
        for g, group in enumerate(groups):
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), alpha=0.8, color=_gcol_1(g), label =f"Grp {g}")
        sm_res = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
        sm_res.set_array([])
        fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title("Set-averaged spectra — all sets")
        plt.tight_layout()
        plt.savefig(f"{warm_run_dir}/set_averaged_spectra_all.png", dpi = 150, bbox_inches='tight')
        plt.close()


        # Plot set average spectra for all sets 3x3
        fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(26, 10))
        axes_flat = axes.flatten()
        groups_per_subplot = 3
        for ax_idx, ax in enumerate(axes_flat):
            start = ax_idx * groups_per_subplot
            end = start + groups_per_subplot
            for g in range(start, min(end, len(groups))):
                ax.plot(
                    np.mean([x[1] for x in groups[g]], axis=0) / 1e6,
                    np.mean([x[0] for x in groups[g]], axis=0),
                    alpha=0.8, color=_gcol_1(g), label=f"Grp {g}")

        for row in range(3):
            axes[row, 0].set_ylabel("PSD  [V²/Hz]")
        for col in range(3):
            axes[2, col].set_xlabel("IF frequency  [MHz]")

        sm_res = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
        sm_res.set_array([])
        fig.colorbar(sm_res, ax=axes_flat, label="Mean cavity resonance  [GHz]")
        # fig.subplots_adjust(wspace=0.05, hspace=0.05)

        fig.canvas.draw()
        positions = [ax.get_position() for ax in axes_flat]
        left = min(p.x0 for p in positions)
        right = max(p.x1 for p in positions)
        center_x = (left + right) / 2

        fig.suptitle("Set-averaged spectra — all sets", fontsize=32, x=center_x)
        plt.savefig(f"{warm_run_dir}/set_averaged_spectra_all_3x3.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot standard deviation of averaged sets againist frequency for all sets 3x3
        fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(26, 10))
        axes_flat = axes.flatten()
        groups_per_subplot = 3
        for ax_idx, ax in enumerate(axes_flat):
            start = ax_idx * groups_per_subplot
            end = start + groups_per_subplot
            for g in range(start, min(end, len(groups))):
                ax.plot(
                    np.mean([x[1] for x in groups[g]], axis=0) / 1e6,
                    np.std([x[0] for x in groups[g]], axis=0),
                    alpha=0.8, color=_gcol_1(g), label=f"Grp {g}")

        for row in range(3):
            axes[row, 0].set_ylabel("Standard Deviation  [V²/Hz]")
        for col in range(3):
            axes[2, col].set_xlabel("IF frequency  [MHz]")

        sm_res = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
        sm_res.set_array([])
        fig.colorbar(sm_res, ax=axes_flat, label="Mean cavity resonance  [GHz]")

        fig.canvas.draw()
        positions = [ax.get_position() for ax in axes_flat]
        left = min(p.x0 for p in positions)
        right = max(p.x1 for p in positions)
        center_x = (left + right) / 2

        fig.suptitle("Standard deviation of averaged spectra againist frequency - all sets", fontsize=32, x=center_x)
        plt.savefig(f"{warm_run_dir}/std_vs_freq_all_3x3.png", dpi=150, bbox_inches='tight')
        plt.close()
        

        # Plot standard deviation of averaged sets againist frequency for all sets
        fig, ax = plt.subplots(figsize=(13, 7))
        for g, group in enumerate(groups):
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.std([x[0] for x in group], axis=0), alpha=0.8, color=_gcol_1(g), label =f"Grp {g}")
        sm_res = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
        sm_res.set_array([])
        fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("Standard deviation  [V²/Hz]")
        ax.set_title("Standard deviation of averaged spectra againist frequency - all sets")
        plt.tight_layout()
        plt.savefig(f"{warm_run_dir}/std_vs_freq_all.png", dpi = 150, bbox_inches='tight')
        plt.close()


        # Plot average std for each group againist set number
        av_stds = []
        for g, group in enumerate(groups):
            std = np.std([x[0] for x in group], axis=0)
            av_stds.append(np.mean(std))
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.scatter(range(0, len(groups)), av_stds)
        ax.set_xlabel("Set number")
        ax.set_ylabel("Standard deviation  [V²/Hz]")
        ax.set_title("Average standard deviation per set againist set number")
        plt.tight_layout()
        plt.savefig(f"{warm_run_dir}/std_vs_set_num.png", dpi = 150, bbox_inches='tight')
        plt.close()


        for g, group in enumerate(tqdm(groups, desc="Set averaging diagnostic plots")):
        # for g, (freqs, specs) in enumerate(group_avg_spectra):

            # Plot set averaged spectra + the sets spectra per set
            fig, ax = plt.subplots(figsize=(13, 7))
            greys = cm.Greys(np.linspace(0.3, 0.9, len(group)))
            for i, x in enumerate(group):
                ax.plot(x[1]/1e6, x[0], color=greys[i])
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), alpha=0.8, color="red", label="set averaged")
            # ax.plot(freqs/1e6, specs, alpha=0.8, color="red", label="set averaged")
            norm = mcolors.Normalize(vmin=0, vmax=len(group))
            sm = ScalarMappable(cmap=cm.Greys, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Spectrum index in set")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"Set-averaged spectra and the individual spectra — set {g}")
            plt.tight_layout()
            plt.savefig(f"{warm_run_dir}/set_and_average_spectra_{g}.png", dpi = 150, bbox_inches='tight')
            plt.close()


            # Plot set averaged spectra + the sets spectra per group - log plot
            fig, ax = plt.subplots(figsize=(13, 7))
            greys = cm.Greys(np.linspace(0.3, 0.9, len(group)))
            for i, x in enumerate(group):
                ax.plot(x[1]/1e6, np.log(x[0]), color=greys[i])
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.log(np.mean([x[0] for x in group], axis=0)), alpha=0.8, color="red", label="set averaged")
                # ax.plot(x[1]/1e6, (x[0]), color=greys[i])
            # ax.plot(freqs/1e6, (specs), alpha=0.8, color="red", label="set averaged")
            norm = mcolors.Normalize(vmin=0, vmax=len(group))
            sm = ScalarMappable(cmap=cm.Greys, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Spectrum index in set")
            # ax.set_yscale("log")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"log set-averaged spectra and the log individual spectra — set {g}")
            plt.tight_layout()
            plt.savefig(f"{warm_run_dir}/log_set_and_average_spectra_{g}.png", dpi = 150, bbox_inches='tight')
            plt.close()


            # Plot set averaged spectra with errors per group
            fig, ax = plt.subplots(figsize=(13, 7))
            # ax.errorbar(freqs/1e6, specs, alpha=0.25, ecolor="blue", color="red")
            # ax.plot(freqs/1e6, specs, alpha=0.8, color='red')
            ax.errorbar(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), np.std([x[0] for x in group], axis=0), alpha=0.25, ecolor="blue", color="red")
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), alpha=0.8, color='red')
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"Set-averaged spectra with errors — set {g}")
            ax.set_yscale("log")
            plt.tight_layout()
            plt.savefig(f"{warm_run_dir}/set_averaged_spectra_errors_{g}.png", dpi = 150, bbox_inches='tight')
            plt.close()


            # Plot zoomed set averaged spectra with errors zoomed in per group
            fig, ax = plt.subplots(figsize=(13, 5))

            freqs_avg = np.mean([x[1] for x in group], axis=0) / 1e6
            spec_avg = np.mean([x[0] for x in group], axis=0)
            spec_std = np.std([x[0] for x in group], axis=0)

            ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.5, ecolor="blue", color="red")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"Set-averaged spectra with errors — set {g} (zoomed)")

            x_min, x_max = 1.5, 1.75
            ax.set_xlim(x_min, x_max)

            in_range = (freqs_avg >= x_min) & (freqs_avg <= x_max)
            if in_range.any():
                y_lower = np.min(spec_avg[in_range] - spec_std[in_range])
                y_upper = np.max(spec_avg[in_range] + spec_std[in_range])
                y_pad = 0.05 * (y_upper - y_lower)
                ax.set_ylim(y_lower - y_pad, y_upper + y_pad)

            plt.tight_layout()
            plt.savefig(f"{warm_run_dir}/set_averaged_spectra_errors_zoom{g}.png", dpi=150, bbox_inches='tight')
            plt.close()
        

            # Plot histogram of each set averaged spectra per group
            fig,ax = plt.subplots(figsize=(9, 5))
            # ax.hist(specs, bins=100)
            ax.hist(np.mean([x[0] for x in group], axis=0), bins=100)
            ax.set_xlabel("PSD  [V²/Hz]")
            ax.set_ylabel("Counts")
            ax.set_title(f"Histogram of set averaged group {g}")
            plt.axvline(x = np.mean([x[0] for x in group]), color = 'r', alpha=0.8, ls="--", label = f'mean value ({np.mean([x[0] for x in group])})')
            plt.axvline(x = np.median([x[0] for x in group]), color = 'g', alpha=0.8, ls="--", label = f'median value ({np.median([x[0] for x in group])})')
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"{warm_run_dir}/histogram_of_group_{g}", dpi = 150, bbox_inches='tight')
            plt.close()


            # Plot the standard deviation of each set againist set number per group
            set_std = np.std([x[0] for x in group], axis=0)
            set_num = range(0, len(set_std))
            fig,ax = plt.subplots(figsize=(13,7))
            ax.scatter(set_num, set_std, alpha=0.8)
            ax.set_xlabel("Set number")
            ax.set_ylabel("Standard deviation [V²/Hz]")
            ax.set_title(f"Standard deviation againist set number — set {g}")
            plt.tight_layout()
            plt.savefig(f"{warm_run_dir}/std_againist_set_{g}.png", dpi = 150, bbox_inches='tight')
            plt.close()


            # Plot standard deviation of each set average againist frequency per group
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.std([x[0] for x in group], axis=0), alpha=0.8, color=_gcol_1(g), label =f"Grp {g}")
            sm_res = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
            sm_res.set_array([])
            fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("Standard deviation  [V²/Hz]")
            ax.set_title(f"Standard deviation of set average againist frequency - set {g}")
            plt.tight_layout()
            plt.savefig(f"{warm_run_dir}/std_vs_freq_{g}.png", dpi = 150, bbox_inches='tight')
            plt.close()
        

    # Plot group averaged spectra with SG fits
    fig, ax = plt.subplots(figsize=(13, 7))
    for g, ((freqs, specs), fit) in enumerate(zip(group_avg_spectra, group_sg_fits)):
        ax.plot(freqs/1e6, specs,lw=1.0, alpha=0.55, color=_gcol_1(g), label=f"Grp {g}")
        ax.plot(freqs/1e6, fit, lw=1.8, alpha=0.95, color=_gcol_1(g), linestyle="--")
    sm_res2 = ScalarMappable(cmap=_cmap_g_1, norm=_norm_res)
    sm_res2.set_array([])
    fig.colorbar(sm_res2, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Group-averaged spectra with initial SG fits  (dashed = fit)")
    plt.tight_layout()
    plt.savefig(f"{warm_run_dir}/group_averaged_spectra_with_sg_fits.png", dpi = 150, bbox_inches='tight')
    plt.close()


    # -----------------------------------------------------------------------
    # Iterative Sigma Clipping
    # -----------------------------------------------------------------------

    if base["clipping_mode"] == "Claude":
        group_masks = [
            np.zeros(len(avg[0]), dtype=int) if avg is not None else None
            for avg in group_avg_spectra
        ]
    elif base["clipping_mode"] == "Blue":
        group_masks = [
            [np.zeros(len(item[0]), dtype=int) for item in group]
            for group in groups
        ]


    sigma_cut = base["sigma_cut"]
    n_iterations = base["n_iterations"]
    for iteration in range(1, n_iterations + 1):
        print(f"\n  --- Iteration {iteration} / {n_iterations} ---")

        if base["clipping_mode"] == "Claude":
            group_masks, group_sg_fits = claude_clipping(
                group_avg_spectra, group_masks, group_sg_fits,
                sigma_cut, base["sg_window_warm"], base["sg_poly_warm"], iteration
                )
            plotting_group_masks = [
                [mask] * len(group) if mask is not None else None
                for mask, group in zip(group_masks, groups)
            ]

        elif base["clipping_mode"] == "Blue":
            group_masks, group_sg_fits = blue_clipping(
                groups, group_masks, group_sg_fits, sigma_cut,
                base["sg_window_warm"], base["sg_poly_warm"], iteration
                )
            plotting_group_masks = group_masks

        fig, ax = plt.subplots(figsize=(14, 5))
        for g, avg in enumerate(group_avg_spectra):
            if avg is None:
                continue
            freqs, specs  = avg
            masks  = plotting_group_masks[g]
            fit   = group_sg_fits[g]

            for spectra, mask in zip(group, masks):
                unmasked = mask == 0
                masked_this_iteration = mask == iteration
                masked_previously = (mask > 0) & (mask != iteration)
                if masked_this_iteration.any():
                    ax.scatter(freqs[masked_this_iteration]/1e6, specs[masked_this_iteration], marker = ".", color=_gcol_2(g), zorder=5)

                if masked_previously.any():
                    ax.scatter(freqs[masked_previously]/1e6, specs[masked_previously], c="grey", zorder=4)

            ax.plot(freqs[unmasked]/1e6, specs[unmasked],lw=1.0, alpha=0.55, color=_gcol_1(g), label=f"Grp {g}")
            ax.plot(freqs/1e6, fit, lw=1.8, alpha=0.95, color=_gcol_1(g), linestyle="--")

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
        plt.savefig(f"{warm_run_dir}/masked_bin_iteration_{iteration}.png", dpi=150, bbox_inches='tight')
        plt.close()


    # --------------------
    # Residuals Plotting
    # --------------------

    if out["clipping_residuals"]:

        clip_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /'clipping_plots'
        clip_run_dir.mkdir(parents=True, exist_ok=True)

        for g, fit in enumerate(tqdm(group_sg_fits, desc="Clipping residuals plots")):

            if base["clipping_mode"] == "Claude":

                avg = group_avg_spectra[g]
                if avg is None or fit is None:
                    continue
                freqs, specs = avg
                residuals = specs - fit

                # Plot residuals againist frequency
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot( freqs/1e6 ,residuals)
                ax.set_xlabel("IF frequency  [MHz]")
                ax.set_ylabel("Residuals  [V²/Hz]")
                ax.set_title(f"Residuals - group {g} (Claude's clipping method)")
                plt.tight_layout()
                plt.savefig(f"{clip_run_dir}/claude_residuals_{g}.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Plot histogram of residuals
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(residuals[np.isfinite(residuals)], bins=50, color="steelblue")
                ax.set_xlabel("Residuals  [V²/Hz]")
                ax.set_ylabel("Counts")
                ax.set_title(f"Residuals histogram - group {g} (Claude's clipping method)")
                plt.tight_layout()
                plt.savefig(f"{clip_run_dir}/claude_residuals_hist_{g}.png", dpi=150, bbox_inches='tight')
                plt.close()


            elif base["clipping_mode"] == "Blue":

                group = groups[g]
                if fit is None or len(group) == 0:
                    continue

                n = len(group)
                colors = cm.viridis(np.linspace(0, 1, n))
                all_residuals = []

                # Plot residuals againist frequnecy for each group
                fig, ax = plt.subplots(figsize=(14, 5))
                for spec_idx, (spectra, frequencies, res_freq) in enumerate(group):
                    residuals = spectra - fit
                    all_residuals.append(residuals)
                    ax.plot(frequencies / 1e6, residuals, lw=0.8, alpha=0.7, color=colors[spec_idx])


                norm = mcolors.Normalize(vmin=0, vmax=n - 1)
                sm = ScalarMappable(cmap=cm.viridis, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label="Spectrum index in group")
                ax.set_xlabel("IF frequency  [MHz]")
                ax.set_ylabel("Residuals  [V²/Hz]")
                ax.set_title(f"Residuals — group {g} (Blue's clipping method)")
                plt.tight_layout()
                plt.savefig(f"{clip_run_dir}/blue_residuals_{g}.png", dpi=150, bbox_inches='tight')
                plt.close()


                # Plot stacked histogram of residuals in each group
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist([r[np.isfinite(r)] for r in all_residuals], bins=50, stacked=True, color=colors)
                norm2 = mcolors.Normalize(vmin=0, vmax=n - 1)
                sm2 = ScalarMappable(cmap=cm.viridis, norm=norm2)
                sm2.set_array([])
                fig.colorbar(sm2, ax=ax, label="Spectrum index in group")
                ax.set_xlabel("Residuals  [V²/Hz]")
                ax.set_ylabel("Counts")
                ax.set_title(f"Residuals histogram (stacked) — group {g} (Blue's clipping method)")
                plt.tight_layout()
                plt.savefig(f"{clip_run_dir}/blue_residuals_hist_{g}.png", dpi=150, bbox_inches='tight')
                plt.close()

    specs, fper = finalise_specs(base["clipping_mode"], group_avg_spectra, groups, group_sg_fits)


    # ===================================
    # Varying Group Size
    # ====================================

    specs_grp = shifted_spectra

    if out["varying_group_size"]:

        var_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' / 'varying_group_size'
        var_dir.mkdir(parents=True, exist_ok=True)

        spacings_config = [5, 10, 15, 30, 60, 90, 120, 150, 180, 210, 240]  # minutes
        n_total = len(dts)


        def build_groups(spacing_minutes):
            spacing_sec = float(spacing_minutes * 60)
            groups_out = []
            i = 0
            while i < n_total:
                j = i + 1
                while j < n_total and (dts[j] - dts[i]).total_seconds() < spacing_sec:
                    j += 1
                groups_out.append([[specs_grp[k], fper[k], metadata["res_freq"][k]] for k in range(i, j)])
                i = j
            return groups_out


        groups_by_spacing = {sp: build_groups(sp) for sp in spacings_config}

        var_results = []

        for spacing in spacings_config:
            var_groups = groups_by_spacing[spacing]
            group_sizes = [len(g) for g in var_groups]
            residual_stds = []
            residual_avgs = []

            group_avg_spectra_var = []
            group_sg_fits_var = []
            for group in var_groups:

                freqs_avg = np.mean([x[1] for x in group], axis=0)
                spec_avg = np.mean([x[0] for x in group], axis=0)
                group_avg_spectra_var.append((freqs_avg, spec_avg))

                if not spec_avg.any():
                    continue
                try:
                    _, baseline = remove_baseline(
                        spectrum=spec_avg,
                        window_length=base["sg_window_warm"],
                        polyorder=base["sg_poly_warm"],
                    )
                    group_sg_fits_var.append(baseline)

                except Exception as e:
                    print(f"[Grouping variation] spacing={spacing}minutes: SG fit failed ({e}), skipping group")
                    group_sg_fits_var.append(None)


            if base["clipping_mode"] == "Claude":
                group_masks_var = [
                    np.zeros(len(avg[0]), dtype=int) if avg is not None else None
                    for avg in group_avg_spectra_var
                ]
            elif base["clipping_mode"] == "Blue":
                group_masks_var = [
                    [np.zeros(len(item[0]), dtype=int) for item in group]
                    for group in var_groups
                ]
            
            for iteration in range(1, n_iterations + 1):
                if base["clipping_mode"] == "Claude":
                    group_masks_var, group_sg_fits_var = claude_clipping(
                        group_avg_spectra_var, group_masks_var, group_sg_fits_var,
                        sigma_cut, base["sg_window_warm"], base["sg_poly_warm"], iteration
                    )
                elif base["clipping_mode"] == "Blue":
                    group_masks_var, group_sg_fits_var = blue_clipping(
                        var_groups, group_masks_var, group_sg_fits_var, sigma_cut,
                        base["sg_window_warm"], base["sg_poly_warm"], iteration
                )

            if base["clipping_mode"] == "Claude":
                total_masked = sum(int(np.count_nonzero(m)) for m in group_masks_var if m is not None)
                total_bins = sum(len(m) for m in group_masks_var if m is not None)
            elif base["clipping_mode"] == "Blue":
                total_masked = sum(int(np.count_nonzero(m)) for gm in group_masks_var for m in gm)
                total_bins = sum(len(m) for gm in group_masks_var for m in gm)

            fraction_masked = (total_masked / total_bins)


            if base["clipping_mode"] == "Claude":
                for avg, mask, fit in zip(group_avg_spectra_var, group_masks_var, group_sg_fits_var):
                    if avg is None or mask is None or fit is None:
                        continue
                    _, spec_avg = avg
                    unmasked = mask == 0
                    if not unmasked.any():
                        continue
                    group_residuals = spec_avg[unmasked] - fit[unmasked]
                    residual_stds.append(np.nanstd(group_residuals))
                    residual_avgs.append(np.nanmean(group_residuals))

            elif base["clipping_mode"] == "Blue":
                for group, group_mask, fit in zip(var_groups, group_masks_var, group_sg_fits_var):
                    if fit is None or len(group) == 0:
                        continue
                    for (spectra, frequencies, res_freq), mask in zip(group, group_mask):
                        unmasked = mask == 0
                        if not unmasked.any():
                            continue
                        residuals_i = spectra[unmasked] - fit[unmasked]
                        residual_stds.append(np.nanstd(residuals_i))
                        residual_avgs.append(np.nanmean(residuals_i))


            var_results.append({
                "spacing_minutes": spacing,
                "n_groups": len(var_groups),
                "average_group_size": np.mean(group_sizes),
                "average_residual_std": np.mean(residual_stds),
                "average_residual_average": np.mean(np.absolute(np.array(residual_avgs))),
                "total_masked": total_masked, 
                "total_bins": total_bins,
                "fraction_masked": fraction_masked,
            })


        # ------------------
        # Plotting
        # ------------------

        spacings_plot = [r["spacing_minutes"] for r in var_results]
        n_groups      = [r["n_groups"] for r in var_results]
        grp_size      = [r["average_group_size"] for r in var_results]
        resid_av      = [r["average_residual_average"] for r in var_results]
        resid_std     = [r["average_residual_std"] for r in var_results]
        fractions_masked = [r["fraction_masked"] for r in var_results]
        total_masked = [r["total_masked"] for r in var_results]


        # Plot absolute residual average vs grouping size
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(spacings_plot, resid_av, marker="o", alpha=0.7)
        ax.set_xlabel("Grouping time [minutes]")
        ax.set_ylabel("Average residuals  [V²/Hz]")
        ax.set_title(f"SG fit residual mean vs. grouping time ({base['clipping_mode']} clipping mode)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{var_dir}/residual_avg_vs_grouping.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot residual std vs grouping size
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(spacings_plot, resid_std, marker="o")
        ax.set_xlabel("Grouping time threshold  [minutes]")
        ax.set_ylabel("Average residual std  [V²/Hz]")
        ax.set_title(f"Average residual std vs. grouping time ({base['clipping_mode']} clipping mode)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{var_dir}/residual_std_vs_grouping.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot SG fit for first group for each group size
        fig, ax = plt.subplots(figsize=(13, 7))
        colors = cm.viridis(np.linspace(0, 1, len(spacings_config)))

        for c_idx, test_spacing in enumerate(spacings_config):
            var_groups = groups_by_spacing[test_spacing]
            rep_group = next((g for g in var_groups if len(g) > 0), None)
            if rep_group is None:
                continue

            freqs_avg = np.mean([x[1] for x in rep_group], axis=0)
            spec_avg = np.mean([x[0] for x in rep_group], axis=0)
            try:
                _, baseline = remove_baseline(
                    spectrum=spec_avg,
                    window_length=base["sg_window_warm"],
                    polyorder=base["sg_poly_warm"],
                )
            except Exception as e:
                print(f"[SG overlay] spacing={test_spacing}minutes: SG fit failed ({e}), skipping")
                continue

            ax.plot(freqs_avg / 1e6, baseline, color=colors[c_idx],
                    label=f"{test_spacing} min  (n={len(rep_group)})")

        norm_spacing = mcolors.Normalize(vmin=min(spacings_config), vmax=max(spacings_config))
        sm_res = ScalarMappable(cmap=cm.viridis, norm=norm_spacing)
        sm_res.set_array([])
        fig.colorbar(sm_res, ax=ax, label="Grouping time  [minutes]")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title("Group averaged spectra for grouping time variation (1st group)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{var_dir}/sg_fit_grouping_time.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot per-group size diagnostic plots: set + average, set + average with errors, zoomed
        for test_spacing in tqdm(spacings_config, desc="Group size variation diagnostic plots"):

            var_groups = groups_by_spacing[test_spacing]
            rep_group = next((g for g in var_groups if len(g) > 0), None)

            if rep_group is None:
                continue

            freqs_avg = np.mean([x[1] for x in rep_group], axis=0) / 1e6
            spec_avg = np.mean([x[0] for x in rep_group], axis=0)
            spec_std = np.std([x[0] for x in rep_group], axis=0)


            # Plot set + average spectra 
            fig, ax = plt.subplots(figsize=(13, 7))
            greys = cm.Greys(np.linspace(0.3, 0.9, len(rep_group)))
            for i_spec, x in enumerate(rep_group):
                ax.plot(x[1] / 1e6, x[0], color=greys[i_spec])
            ax.plot(freqs_avg, spec_avg, alpha=0.8, color="red", label="set averaged")
            norm = mcolors.Normalize(vmin=0, vmax=len(rep_group))
            sm = ScalarMappable(cmap=cm.Greys, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Spectrum index in set")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"Set-averaged spectra and individual spectra — spacing {test_spacing} minutes (n={len(rep_group)})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{var_dir}/set_and_average_spectra_spacing_{test_spacing}.png", dpi=150, bbox_inches='tight')
            plt.close()


            # Plot average spectra with errors
            fig, ax = plt.subplots(figsize=(13, 7))
            ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.10, ecolor="blue", color="red")
            ax.plot(freqs_avg, spec_avg, alpha=1, color="red", label="set averaged")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"Set-averaged spectra with errors — spacing {test_spacing} minutes (n={len(rep_group)})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{var_dir}/average_spectra_errors_spacing_{test_spacing}.png", dpi=150, bbox_inches='tight')
            plt.close()


            # Plot average spectra with errors, zoomed
            fig, ax = plt.subplots(figsize=(26, 14))
            ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.5, ecolor="blue", color="red")
            ax.set_xlabel("IF frequency  [MHz]")
            ax.set_ylabel("PSD  [V²/Hz]")
            ax.set_title(f"Set-averaged spectra with errors zoomed - spacing {test_spacing} min (n={len(rep_group)})")

            x_min, x_max = 1.5, 2
            ax.set_xlim(x_min, x_max)
            in_range = (freqs_avg >= x_min) & (freqs_avg <= x_max)
            if in_range.any():
                y_lower = np.min(spec_avg[in_range] - spec_std[in_range])
                y_upper = np.max(spec_avg[in_range] + spec_std[in_range])
                y_pad = 0.05 * (y_upper - y_lower)
                ax.set_ylim(y_lower - y_pad, y_upper + y_pad)

            plt.tight_layout()
            plt.savefig(f"{var_dir}/average_spectra_errors_zoom_spacing_{test_spacing}.png", dpi=150, bbox_inches='tight')
            plt.close()


        # Plot avg-of-avg std vs group size
        spacing_avg_of_avg_std = []
        for spacing in spacings_config:
            var_groups = groups_by_spacing[spacing]
            group_avg_stds = [np.mean(np.std([x[0] for x in group], axis=0)) for group in var_groups if len(group) > 0]
            spacing_avg_of_avg_std.append(np.mean(group_avg_stds) if group_avg_stds else np.nan)

        fig, ax = plt.subplots(figsize=(13, 7))
        ax.plot(spacings_config, spacing_avg_of_avg_std, marker="o")
        ax.set_xlabel("Group spacing (minutes)")
        ax.set_ylabel("Average standard deviation  [V²/Hz]")
        ax.set_title("Average standard deviation vs group spacing")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{var_dir}/avg_avg_std_vs_group_spacing.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot average cavity resonance drift within a group againist group size
        spacing_avg_res_spread = []
        for spacing in spacings_config:
            var_groups = groups_by_spacing[spacing]
            res_spreads_this = []

            for group in var_groups:
                if len(group) < 2:
                    continue 
                res_freqs_in_group = [item[2] for item in group]
                res_freqs_in_group = np.asarray(res_freqs_in_group, dtype=float)
                finite_vals = res_freqs_in_group[np.isfinite(res_freqs_in_group)]
                if len(finite_vals) < 2:
                    continue
                spread = np.max(finite_vals) - np.min(finite_vals)
                res_spreads_this.append(spread)

            if len(res_spreads_this) == 0:
                spacing_avg_res_spread.append(np.nan)
                continue

            spacing_avg_res_spread.append(np.mean(res_spreads_this))

        fig, ax = plt.subplots(figsize=(13, 7))
        ax.plot(spacings_config, spacing_avg_res_spread, marker="o")
        ax.set_xlabel("Grouping time  [minutes]")
        ax.set_ylabel("Average resonance frequency spread  [GHz]")
        ax.set_title("Cavity resonance drift within a group vs. grouping time")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{var_dir}/resonance_drift_vs_grouping.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot total bins masked againist grouping time
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.plot(spacings_config, total_masked, marker="o")
        ax.set_xlabel("Grouping time  [minutes]")
        ax.set_ylabel("Total masked bins ")
        ax.set_title(f"Total masked bins vs. grouping time ({base['clipping_mode']} clipping mode)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{var_dir}/total_masked_vs_grouping.png", dpi=150, bbox_inches='tight')
        plt.close()


        print("\n[Grouping time variation summary]")
        for r in var_results:
            print(f"  grouping time={r['spacing_minutes']:>4} min | "
                f"n_groups={r['n_groups']:>4} | "
                f"average size={r['average_group_size']:.1f} | "
                f"average residual std={r['average_residual_std']:.4g} | "
                f"masked={r['total_masked']:>6}/{r['total_bins']:<6}")



    # ---------
    # Old Code
    # ---------

    colour_vals = np.abs(cw_freqs - res_freqs*1e9) / 1e9  # Hz -> GHz

    cbar_label  = r"$|f_{\rm CW} - f_{\rm res}|$  [GHz]"
    cmap        = plt.cm.inferno
    _finite     = colour_vals[np.isfinite(colour_vals)]
    norm = Normalize(
        vmin=np.percentile(_finite,  0),
        vmax=np.percentile(_finite, 100),
    )

    fig, ax = plt.subplots(figsize = (12,6))
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
    plt.figure(figsize=(13, 7))
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
    plt.figure(figsize=(13, 7))
    plt.plot(freqs_r/1e9, z, lw=0.8)
    plt.title("Grand spectrum z-score (SHM matched filter)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("z"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run_dir/"grand_z.png", dpi=150); plt.close()

    # 5) candidates
    ex_run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}' /'candidates_and_exclusion'
    ex_run_dir.mkdir(parents=True, exist_ok=True)

    theta = threshold_for_detection(det["target_snr"], det["confidence"])
    cands, _ = find_candidates(Dg, sg, theta, min_separation=K-1)
    # After: cands, z = find_candidates(Dg, sg, theta, min_separation=K-1)

    fig, ax = plt.subplots(figsize=(13, 7))

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
    fig.savefig(ex_run_dir/"candidates.png", dpi=150)
    plt.close(fig)


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

    # 6) exclusion

    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=det["target_snr"], g0=det["g0"], snr_efficiency=det["snr_eff"])
    plot_exclusion(freqs_r, gmin, outfile=ex_run_dir/"exclusion.png", title="95% CL Exclusion (SHM)")
    with (ex_run_dir/"exclusion.csv").open("w") as fh:
        fh.write("freq_Hz,g_min_rel_to_g0\n")
        for f,g in zip(freqs_r, gmin):
            if np.isfinite(g): fh.write(f"{f},{g}\n")

    print(f"[OK] Run dir: {ex_run_dir}")
    print(f"Candidates flagged: {len(cands)}  (threshold = {theta:.2f}σ)")


if __name__ == "__main__":
    main()