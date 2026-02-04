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
import matplotlib.pyplot as plt
import time
from typing import Sequence, List, Optional, Tuple
import math
import shutil

from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline, align_and_average_spectra
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape  import shm_maxwell_template
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit, plot_exclusion
from axion_haloscope.data_quality import filter_spectrum_set, too_noisy
from axion_haloscope.io import SpectrumSet
from axion_haloscope.data_quality import filter_spectrum_set, too_noisy
from axion_haloscope.io import SpectrumSet
from axion_haloscope.width_fq   import width_from_fq


def _get(d, key, default):
    v = d.get(key, default)
    return default if v is None else v

def load_yaml_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    sim  = raw.get("simulation", {}) or {}
    inj  = raw.get("injection",  {}) or {}
    qc   = raw.get("quality", {}) or {}
    base = raw.get("baseline",   {}) or {}
    rb   = raw.get("rebin",       {}) or {}
    det  = raw.get("detection",   {}) or {}
    out  = raw.get("output",      {}) or {}

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
        "injection": {
            "enabled":     bool(_get(inj, "enabled", False)),
            "f_axion_hz":  inj.get("f_axion_hz", None),  # optional
            "total_power": float(_get(inj, "total_power", 20.0)),
        },
        "baseline": {
            "sg_window": int(_get(base, "sg_window", 401)),
            "sg_poly":   int(_get(base, "sg_poly", 4)),
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
    sim, inj, base, rb, det, out = (cfg[k] for k in ("simulation","injection","baseline","rebin","detection","output"))

    # Output folder
    out_root = pathlib.Path(out["root"])/ "sim_spectra"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f'{out["subdir_prefix"]}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped copies of the config
    cfg_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save exact input YAML as provided
    try:
        stamped_name = f"{cfg_path.stem}_{cfg_stamp}{cfg_path.suffix}"
        shutil.copy(cfg_path, run_dir / stamped_name)
    except Exception as e:
        print(f"[WARN] Could not copy input config file: {e}")

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

    # 1) simulate
    specs, fper, rf, rf_map = simulate_spectra(
        n_spectra=sim["n_spectra"], n_bins=sim["n_bins"],
        bin_width_hz=sim["bin_width_hz"], f_start_hz=sim["f_start_hz"],
        tune_step_bins=sim["tune_step_bins"], rng_seed=sim["rng_seed"],
        noise_sigma=sim["noise_sigma"], axion=ax
    )

    # Always save one example raw spectrum
    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir/"raw_spectrum_first.png", dpi=150); plt.close()
    
    plt.figure(figsize=(9,3))
    plt.plot(fper[-1]/1e9, specs[-1], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example raw spectrum"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir/"raw_spectrum_last.png", dpi=150); plt.close()

    # Optional: save per-spectrum PNGs + spectra.npz
    if out["save_data"]:
        step = max(1, int(out["plots_step"]))
        max_plots = None if out["max_plots"] is None else int(out["max_plots"])
        count = 0
        for i, (freqs, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break
            fig, axp = plt.subplots(figsize=(9,3))
            axp.plot(freqs/1e9, spec, lw=0.6)
            axp.set(xlabel="Frequency [GHz]", ylabel="Raw Power [arb]", title=f"Spectrum {i:03d}")
            axp.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(run_dir / f"spectrum_{i:03d}.png", dpi=120)
            plt.close(fig)
            count += 1
        np.savez(run_dir/"spectra.npz", spectra=np.array(specs), freqs=fper, rf_grid=rf)

        

    t0 = time.time()

    # QC: drop bad spectra (default thresholds; adjust if desired)
    sset = SpectrumSet(spectra=list(specs), freqs_per_spec=list(fper), rf_grid=rf, rf_index_map=list(rf_map))
    sset_qc = sset
    #sset_qc, kept, bad = filter_spectrum_set(sset, predicate=lambda s,f,i: too_noisy(s,f,i, rms_max=3.0))
    #print(f"[QC] kept {len(kept)}/{sset.n_spectra()} spectra; dropped: {bad}")
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map = sset_qc.spectra, sset_qc.freqs_per_spec, sset_qc.rf_grid, sset_qc.rf_index_map


    common_x, padded, average, average_baseline = align_and_average_spectra(fper, specs)

    mode_1 = "additive"
    mode_2 = "additive"
    mode_3 = "multiplicative"


    processed_average, baseline_average = remove_baseline(
    spectrum=average,
    window_length=base["sg_window"] * 10,
    polyorder=base["sg_poly"],
    mode=mode_1,
    subtract_one=False,
    diagnostic={"outfile": run_dir / "1_baseline_of_average_spectra.png",
                "title": "Baseline removal (Average Spectra)"},
    freqs_hz=common_x,
    )

    averaged_baselines = []
    for s in average_baseline:
        processed, _baseline = remove_baseline(
            s,
            window_length=base["sg_window"]* 10,
            polyorder=base["sg_poly"],
            mode=mode_1,
            subtract_one=False,
        )
        averaged_baselines.append(_baseline)



    _= remove_baseline(
    spectrum=specs[0],
    window_length=base["sg_window"],
    polyorder=base["sg_poly"],
    mode=mode_2,
    subtract_one=False,
    add_one=(mode_2 == "additive" and mode_3 =="multiplicative"),
    diagnostic={"outfile": run_dir / "2_baseline_removal_using_average_baseline_spectrum.png",
                "title": "Baseline removal (spectrum 0 using average basline)"},
    freqs_hz=fper[0],
    baseline=averaged_baselines[0],
    )

    med_processed = []
    for s,b in zip(specs, averaged_baselines):
        processed, _baseline = remove_baseline(
            s,
            window_length=base["sg_window"],
            polyorder=base["sg_poly"],
            mode=mode_2,
            add_one=(mode_2 == "additive" and mode_3 =="multiplicative"),
            subtract_one=False,
            baseline=b,
        )
        med_processed.append(processed)



    # 2) baseline removal
    _= remove_baseline(
    spectrum=med_processed[0],
    window_length=base["sg_window"],
    polyorder=base["sg_poly"],
    mode=mode_3,
    subtract_one=(mode_3 == "multiplicative"),
    diagnostic={"outfile": run_dir / "3_baseline_removal_of_processed_spectra.png",
                "title": "Baseline removal (spectrum 0)"},
    freqs_hz=fper[0],
    )

    proc = []
    for s in med_processed:
        processed, _baseline = remove_baseline(
            s,
            window_length=base["sg_window"],
            polyorder=base["sg_poly"],
            mode=mode_3,
            subtract_one=(mode_3 == "multiplicative"),
        )
        proc.append(processed)

   

    # 3) combine
    combined, sigma_c, counts = combine_ml(proc, rf_map, total_rf_bins=len(rf))
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
    Lq = shm_maxwell_template(K=K, bin_width_hz=C*sim["bin_width_hz"], f0_hz=f0)
    Dg, sg = grand_spectrum_ml(Dr, sr, Lq)

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