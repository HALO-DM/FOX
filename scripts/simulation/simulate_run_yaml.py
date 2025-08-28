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

from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape  import shm_maxwell_template
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit, plot_exclusion


def _get(d, key, default):
    v = d.get(key, default)
    return default if v is None else v

def load_yaml_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    sim = raw.get("simulation", {}) or {}
    inj = raw.get("injection",  {}) or {}
    base = raw.get("baseline",   {}) or {}
    rb  = raw.get("rebin",       {}) or {}
    det = raw.get("detection",   {}) or {}
    out = raw.get("output",      {}) or {}

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
            "sigma_hz":    float(_get(inj, "sigma_hz", 2500.0)),
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

    # Axion injection (center mid-span if not provided)
    ax = None
    if inj["enabled"]:
        total_bins = sim["n_bins"] + (sim["n_spectra"] - 1) * sim["tune_step_bins"]
        f_ax = inj["f_axion_hz"]
        if f_ax is None:
            f_ax = sim["f_start_hz"] + 0.5 * total_bins * sim["bin_width_hz"]
        ax = AxionParams(f_axion_hz=float(f_ax), sigma_hz=inj["sigma_hz"], total_power=inj["total_power"])

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
    plt.savefig(run_dir/"raw_spectrum.png", dpi=150); plt.close()

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

    # 2) baseline removal
    proc = [remove_baseline(s, window_length=base["sg_window"], polyorder=base["sg_poly"], subtract_one=True)[0]
            for s in specs]

    # 3) combine
    combined, sigma_c, counts = combine_ml(proc, rf_map, total_rf_bins=len(rf))
    plt.figure(figsize=(10,3))
    plt.plot(rf/1e9, combined, lw=0.8)
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
