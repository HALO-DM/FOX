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
import matplotlib.pyplot as plt
import time
import pandas as pd
import h5py


from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape  import shm_maxwell_template
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit, plot_exclusion
from axion_haloscope.data_quality import filter_spectrum_set, too_noisy, power_too_high, metadata_is_zeros
from axion_haloscope.io import SpectrumSet, SpectrumMetadata, read_hdf5, write_hdf5, read_qshs_hdf5_dir2
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
            "read_input": bool(_get(inp, "read_input", False)),
            "directory":        _get(inp, "directory", "scripts/qshs/output/qshs_import"),
            "input_file_name":  _get(inp, "input_file_name", "spectra.h5"),
        },
        "injection": {
            "enabled":     bool(_get(inj, "enabled", False)),
            "f_axion_hz":  inj.get("f_axion_hz", None),  # optional
            "total_power": float(_get(inj, "total_power", 20.0)),
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


def main():
    ap = argparse.ArgumentParser(description="Simulate haloscope run from YAML config")
    ap.add_argument("config", help="Path to YAML config (e.g. configs/simulate_run.yaml)")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    cfg = load_yaml_config(cfg_path)
    sim, inp, inj, base, rb, det, out = (cfg[k] for k in ("simulation","input","injection","baseline","rebin","detection","output"))

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
    sset, sset_power, kept, bad_power = filter_spectrum_set(
        sset,
        predicate=lambda s, f, md, i: power_too_high(
            s,
            f,
            md,
            i,
            p_max=1e-8,
        ),
    )

    sset, sset_noise, kept, bad_noise = filter_spectrum_set(
        sset,
        predicate=lambda s, f, md, i,: too_noisy(
            s,
            f,
            md,
            i,
            rms_max=3.0,
        ),
    )


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

    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    # Append invalid files list with the new files found
    invalid_files = sset.metadata["invalid_files"]
    bad_zero_power = []
    bad_no_metadata = []

    for f in invalid_files:
        if f[1] == "power spectra is zeros":
            bad_zero_power.append(f)
        elif f[1] == "modefit data is missing":
            bad_no_metadata.append(f)

    for s in bad_power:
        invalid_files.append([sset_power.metadata["file_name"][s], "power is too high", sset_power.metadata["date"][s]])

    for s in bad_noise:
        invalid_files.append([sset_noise.metadata["file_name"][s], "data is too noisy", sset_noise.metadata["date"][s]])
    
    for s in bad_zeros_bandwidth:
        invalid_files.append([sset_zeros_bandwidth.metadata["file_name"][s], "bandwidth data is zeros", sset_zeros_bandwidth.metadata["date"][s]])

    for s in bad_zeros_res_freq:
        invalid_files.append([sset_zeros_res_freq.metadata["file_name"][s], "res_freq data is zeros", sset_zeros_res_freq.metadata["file_name"][s]])

    # Create new SSet with new invalid files list to export
    valid_files = sset.metadata["file_name"]
    input_dir = "input/Feb/All"

    # Ordering invalid files list
    invalid_files_df = pd.DataFrame(data=invalid_files, columns=["File_name", "Reason", "Date_Time"])
    invalid_files_df["Date_Time"] = pd.to_datetime(invalid_files_df["Date_Time"], format="%Y-%m-%d %H:%M:%S")
    invalid_files_df = invalid_files_df.sort_values(by="Date_Time")
    invalid_files = invalid_files_df.values.tolist()

    sset_export = read_qshs_hdf5_dir2(
        input_dir,
        valid_files,
        invalid_files,
        pattern="*.hdf5",
        use_shifted_frequency=True,
        sort_frequency=True,
        run_dir=run_dir,
    )

    out_h5 = f"{run_dir}/final_converted_spectra.h5"
    write_hdf5(sset_export, out_h5)
    print(f"[QSHS] Final SpectrumSet saved to: {out_h5}")


    print(f"[QC]: {len(invalid_files)} / {len(kept) +len(invalid_files)} files are invalid.")
    print(f"[QC]: {len(bad_no_metadata)} spectra were removed as files were missing metadata.")
    print(f"[QC]: {len(bad_zero_power)} spectra were removed as power data were arrays of zeros.")
    print(f"[QC]: {len(bad_power)} spectra were removed as power is too high.")
    print(f"[QC]: {len(bad_noise)} spectra were removed as too noisy.")
    print(f"[QC]: {len(bad_zeros_bandwidth)} spectra were removed as bandwidth were arrays of zeros.")
    print(f"[QC]: {len(bad_zeros_res_freq)} spectra were removed as res_freq were arrays of zeros.")
    print(f"{len(kept)} / {len(kept) + len(invalid_files)} files are valid and suitable for anaylsis.")
    
    # =======================================================================
    # Spectra Plotting
    # =======================================================================

    # Seperate invalid spectra to plot
    specs_invalid_power, fper_invalid_power, _, _, _ = sset_power.spectra, sset_power.freqs_per_spec, sset_power.rf_grid, sset_power.rf_index_map, sset_power.metadata

    # Calculate difference in resonant frequency of the cavity between the spectra
    res_freq_diff = []
    for f in metadata["res_freq"]:
        difference = f - metadata["res_freq"][0]
        res_freq_diff.append(difference)

    # Always save one valid example raw spectrum
    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir/"valid_raw_spectrum.png", dpi=150); plt.close()

    plt.figure(figsize=(9,3))
    plt.plot(fper[-1]/1e9, specs[-1], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir/"valid_raw_spectrum_last.png", dpi=150); plt.close()

    step = max(1, int(out["plots_step"]))
    max_plots = None if out["max_plots"] is None else int(out["max_plots"])

    # Save one invalid example raw spectrum if it exists
    if len(specs_invalid_power) != 0:
        plt.figure(figsize=(9,3))
        plt.plot(fper_invalid_power[0]/1e9, specs_invalid_power[0], lw=0.6)
        plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
        plt.title("Example invalid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir/"invalid_raw_spectrum.png", dpi=150); plt.close()

        plt.figure(figsize=(9,3))
        plt.plot(fper_invalid_power[-1]/1e9, specs_invalid_power[-1], lw=0.6)
        plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
        plt.title("Example invalid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir/"invalid_raw_spectrum_last.png", dpi=150); plt.close()

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
            fig.savefig(run_dir / f"spectrum_{i:03d}.png", dpi=120)
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
        plt.savefig(run_dir/"raw_valid_spectrum_all.png", dpi=150); plt.close()

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
            plt.savefig(run_dir/"raw_invalid_spectrum_all.png", dpi=150); plt.close()

    # Optional: plot all valid raw spectra in one figure with offset
    if out["offset_combined_plot"]:
        # Plot the resonance frequency offset againist the spectrum index
        plt.figure(figsize=(9,3))
        plt.scatter( range(len(res_freq_diff)),res_freq_diff)
        plt.xlabel("Spectrum Index"); plt.ylabel("Resonance Frequency Offset [Hz]")
        plt.title("Resonance Frequency Offset vs Spectrum Index"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(run_dir/"res_freq_offset_vs_index.png", dpi=150); plt.close()

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
        plt.savefig(run_dir/"raw_spectrum_all_valid_offset.png", dpi=150); plt.close()  

 
    t0 = time.time()


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
    plt.savefig(run_dir/"trimmed_spectrum_first.png", dpi=150); plt.close()

    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example valid raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir/"trimmed_spectrum_last.png", dpi=150); plt.close()


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
    _cmap_g = plt.cm.viridis



    _group_mean_res = np.array([
    np.nanmean([item[2] for item in group], axis=0)
    for group in groups
    ])
    
    
    from matplotlib.colors import Normalize 
    _finite_res = _group_mean_res[np.isfinite(_group_mean_res)]
    _norm_res = Normalize(
        vmin=np.nanmin(_finite_res) if len(_finite_res) else 0,
        vmax=np.nanmax(_finite_res) if len(_finite_res) else 1,
    )
    def _gcol(g):
        v = _group_mean_res[g]
        if not np.isfinite(v):
            return "grey"
        return _cmap_g(_norm_res(v))
    from matplotlib.cm import ScalarMappable
    fig, ax = plt.subplots(figsize=(13, 5))
    for g, group in enumerate(groups):
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0), alpha=0.8, color=_gcol(g), label =f"Grp {g}")
    sm_res = ScalarMappable(cmap=_cmap_g, norm=_norm_res)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Group-averaged spectra — all groups")
    plt.tight_layout()
    plt.savefig("zgh.png", dpi = 150, bbox_inches='tight')
    plt.close()



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
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, np.mean([x[0] for x in group], axis=0),   lw=1.0, alpha=0.55, color=_gcol(g), label=f"Grp {g}")
        ax.plot(np.mean([x[1] for x in group], axis=0)/1e6, fit, lw=1.8, alpha=0.95, color=_gcol(g), linestyle="--")
    sm_res2 = ScalarMappable(cmap=_cmap_g, norm=_norm_res)
    sm_res2.set_array([])
    fig.colorbar(sm_res2, ax=ax, label="Mean cavity resonance  [GHz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Group-averaged spectra with initial SG fits  (dashed = fit)")
    plt.tight_layout()
    plt.savefig("gay aaa.png", dpi = 150, bbox_inches='tight')
    plt.close()



    sigma_cut = 3.5
    specs = []
    fper = []
    for group in groups:
        fresh_group = []

        for run in range(3):
            fresh_specs = []
            fresh_freqs = []
            average_spectra = np.mean([x[0] for x in group], axis=0)
            sd_spectra = np.std([x[0] for x in group], axis=0)
            _, baseline = remove_baseline(
                spectrum=average_spectra,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
                )

            for spectra, frequencies, _ in group:

                deviation = np.abs(spectra - baseline)
                mask_idx = np.argwhere(deviation > sigma_cut * sd_spectra)
                mask = np.zeros(len(spectra))
                mask[mask_idx] = True

                spec = np.ma.masked_array(spectra, mask)
                freq = np.ma.masked_array(frequencies, mask)

                cleaned_spec = interpolate_nans(spec.filled(np.nan))
                cleaned_freq = interpolate_nans(freq.filled(np.nan))

                if run == 2:
                    cleaned_spec /= baseline
                fresh_specs.append(cleaned_spec)
                fresh_freqs.append(cleaned_freq)

            fresh_group.append((fresh_specs, fresh_freqs))
        
        

        specs.extend(np.array(fresh_group[-1][0]))
        fper.extend(np.array(fresh_group[-1][1]))


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


    # Normalise rf_map
    rf_map_new = []
    for i in rf_map:
        j = i - i[0]
    rf_map_new.append(j)


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