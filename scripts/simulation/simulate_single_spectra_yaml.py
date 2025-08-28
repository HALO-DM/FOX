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
    sim, inj, out = (cfg[k] for k in ("simulation","injection","output"))

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


    print(f"End simulation")


if __name__ == "__main__":
    main()
