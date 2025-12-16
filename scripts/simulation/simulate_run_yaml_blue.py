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


from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
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

    t_sim0 = time.time()
    # Axion injection (center mid-span if not provided)
    ax = None
    if inj["enabled"]:
        total_bins = sim["n_bins"] + (sim["n_spectra"] - 1) * sim["tune_step_bins"]
        f_ax = inj["f_axion_hz"]
        if f_ax is None:
            f_ax = sim["f_start_hz"] + 0.5 * total_bins * sim["bin_width_hz"]
        s_ax = width_from_fq(f_ax)
        ax_params = AxionParams(f_axion_hz=float(f_ax), sigma_hz=s_ax, total_power=inj["total_power"])

    # 1) simulate
    specs, fper, rf, rf_map = simulate_spectra(
        n_spectra=sim["n_spectra"], n_bins=sim["n_bins"],
        bin_width_hz=sim["bin_width_hz"], f_start_hz=sim["f_start_hz"],
        tune_step_bins=sim["tune_step_bins"], rng_seed=sim["rng_seed"],
        noise_sigma=sim["noise_sigma"], axion=ax_params
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

        

    t0 = time.time()

    # QC: drop bad spectra (default thresholds; adjust if desired)
    sset = SpectrumSet(spectra=list(specs), freqs_per_spec=list(fper), rf_grid=rf, rf_index_map=list(rf_map))
    sset_qc, kept, bad = filter_spectrum_set(sset, predicate=lambda s,f,i: too_noisy(s,f,i, rms_max=3.0))
    print(f"[QC] kept {len(kept)}/{sset.n_spectra()} spectra; dropped: {bad}")
    # replace arrays with filtered ones for the rest of the chain
    specs, fper, rf, rf_map = sset_qc.spectra, sset_qc.freqs_per_spec, sset_qc.rf_grid, sset_qc.rf_index_map




    # 2) baseline removal
    _= remove_baseline(
    spectrum=specs[0],
    window_length=base["sg_window"],
    polyorder=base["sg_poly"],
    subtract_one=True,
    diagnostic={"outfile": run_dir / "baseline_s000_before_after.png",
                "title": "Baseline removal (spectrum 0)"},
    freqs_hz=fper[0],
    )

    proc = []
    for s in specs:
        processed, _baseline = remove_baseline(
            s,
            window_length=base["sg_window"],
            polyorder=base["sg_poly"],
            subtract_one=True,
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
    #'''
    # ---------- DEBUG DUMP ----------
    dbg = {}
    dbg["len_Dr"] = len(Dr)
    dbg["len_sr"] = len(sr)
    dbg["len_Lq"] = len(Lq)
    dbg["Lq_min_max"] = (np.nanmin(Lq), np.nanmax(Lq))
    dbg["sr_min_max"] = (np.nanmin(sr), np.nanmax(sr))
    dbg["Lq_median"] = float(np.nanmedian(Lq))
    dbg["sr_median"] = float(np.nanmedian(sr))
    print("[DEBUG] Dr/sr/Lq lengths and mins/maxs:", dbg)

    # plot template and noise and their ratio
    fig, ax = plt.subplots(2,1, figsize=(9,6), sharex=True)
    ax[0].plot(np.arange(len(Lq)), Lq, lw=1, label="Lq (template)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(np.arange(len(sr)), sr, lw=1, label="sr (rebinned noise rms)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir/"debug_Lq_sr.png", dpi=160)
    plt.close(fig)

    # compute a single-centre test Rloc manually at middle index
    n = len(sr)
    K = len(Lq)
    mid = n//2
    r = max(0, min(n-K, mid - K//2))
    segs = sr[r:r+K]
    denom = (Lq*Lq / (segs**2 + 1e-18)).sum()
    print(f"[DEBUG] sample window r={r}: denom={denom}, sqrt={np.sqrt(denom)}")
    # Also check how Rloc scales with an assumed g0 if you scale Lq linearly vs quadratically
    R_linear_like = np.sqrt(((Lq*1.0)**2 / (segs**2 + 1e-18)).sum())
    R_power_like  = np.sqrt((( (Lq**0.5) )**2 / (segs**2 + 1e-18)).sum())  # if Lq was power ~g^2
    print("[DEBUG] example R assuming Lq∝g :", R_linear_like, " assuming Lq∝g^2 (take sqrt):", R_power_like)
    # ---------- end debug ----------
    #'''



    # --- Use injection to measure template scaling (debug / auto-scale) ---
    if inj["enabled"]:
        # re-run pipeline without injection to get baseline grand spectrum
        # (simulate_spectra with ax=None)
        specs_noinj, fper_noinj, rf_noinj, rf_map_noinj = simulate_spectra(
            n_spectra=sim["n_spectra"], n_bins=sim["n_bins"],
            bin_width_hz=sim["bin_width_hz"], f_start_hz=sim["f_start_hz"],
            tune_step_bins=sim["tune_step_bins"], rng_seed=sim["rng_seed"],
            noise_sigma=sim["noise_sigma"]
        )
        # run the same baseline/remove/combine/rebin/grand pipeline on no-injection run:
        proc_noinj = []
        for s in specs_noinj:
            processed, _ = remove_baseline(s, window_length=base["sg_window"],
                                        polyorder=base["sg_poly"], subtract_one=True)
            proc_noinj.append(processed)
        combined_noinj, sigma_c_noinj, _ = combine_ml(proc_noinj, rf_map_noinj, total_rf_bins=len(rf_noinj))
        Dr_noinj, sr_noinj, _ = rebin_ml(combined_noinj, sigma_c_noinj, C=C)
        freqs_r_noinj = rf_noinj[:len(Dr_noinj)*C:C] + (C//2)*sim["bin_width_hz"]
        f0_noinj = freqs_r_noinj[len(freqs_r_noinj)//2]
        Lq_unit = shm_maxwell_template(K=K, bin_width_hz=C*sim["bin_width_hz"], f0_hz=f0_noinj)

        Dg_noinj, sg_noinj = grand_spectrum_ml(Dr_noinj, sr_noinj, Lq_unit)

        # The difference is the recovered injected power per rebinned bin (in the same units as Dg)
        recovered = Dg - Dg_noinj  # shape same as Dg
        # restrict to a small window around the injected f0 to avoid noise bias
        # (choose +/- 2*K rebinned bins)
        center_idx = np.nanargmax(recovered)  # or locate using ax.f_axion_hz
        w = max(3*K, 10)
        window = slice(max(0, center_idx-w), min(len(recovered), center_idx+w))
        recovered_power = np.nansum(recovered[window])  # sum of power recovered (units consistent)
        # normalize Lq_unit and scale to match recovered_power
        Lq_unit_sum = np.nansum(Lq_unit)
        if Lq_unit_sum <= 0:
            raise RuntimeError("Lq unit template normalisation is zero or negative.")
        Lq_scaled = Lq_unit * (recovered_power / Lq_unit_sum)

        # replace Lq with scaled version to be used for limits
        Lq = Lq_scaled
        print(f"[AUTO-SCALE] recovered_power={recovered_power:.3e}; Lq_unit_sum={Lq_unit_sum:.3e}; scale={recovered_power/Lq_unit_sum:.3e}")

    # ------------------ apply the AUTO-SCALE Lq and diagnostics ------------------
    # assume Lq_unit and 'scale' were computed by the AUTO-SCALE block already
    Lq = Lq_unit * (recovered_power / Lq_unit_sum)  # same as Lq_unit * scale

    # compute Rloc and gmin
    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=det["target_snr"], g0=det["g0"], snr_efficiency=det["snr_eff"])

    # stats
 
    finite = np.isfinite(Rloc) & (Rloc > 0)
    print("[SCALE] scale =", float(recovered_power / Lq_unit_sum))
    print("[SCALE] Rloc: median {:.3f}, min {:.3f}, max {:.3f}".format(np.nanmedian(Rloc[finite]), np.nanmin(Rloc[finite]), np.nanmax(Rloc[finite])))
    print("[SCALE] gmin: median {:.3e}, 10p {:.3e}, 90p {:.3e}".format(np.nanmedian(gmin[finite]), np.nanpercentile(gmin[finite],10), np.nanpercentile(gmin[finite],90)))

    # linearity test: R should scale linearly when Lq scaled
    R1 = compute_local_snr_template(sr, Lq)
    R2 = compute_local_snr_template(sr, 2.0 * Lq)
    ratio_med = np.nanmedian(R2[finite] / R1[finite])
    print("[LINEARITY] median R(2*Lq)/R(Lq) =", ratio_med, " (expect ~2.0 )")

    # diagnostic plots: Lq vs recovered and gmin
    fig, ax = plt.subplots(3,1,figsize=(10,9), sharex=True)
    ax[0].plot(freqs_r/1e9, np.tile(np.sum(Lq), len(freqs_r)), lw=0.6, label="Lq total (for reference)")
    ax[0].plot(freqs_r/1e9, np.repeat(0.0, len(freqs_r)), alpha=0)  # dummy for axes
    ax[0].set(title="Template Lq (summed = {:.3g})".format(np.nansum(Lq)), ylabel="(pipeline units)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # plot the normalized template centered at f0
    center = len(freqs_r)//2
    half = len(Lq)//2
    template_x = freqs_r[center-half:center-half+len(Lq)]/1e9
    ax[1].plot(freqs_r/1e9, Dg - np.nanmedian(Dg), lw=0.6, label="grand-spectrum (median sub)")
    ax[1].scatter(template_x, Lq, color="tab:red", label="Lq scaled (over central bins)")
    ax[1].set(ylabel="Dg (arb)"); ax[1].legend(); ax[1].grid(alpha=0.3)

    # gmin plot (log y)
    ax[2].plot(freqs_r/1e9, gmin, lw=0.8)
    ax[2].set(xlabel="Frequency [GHz]", ylabel="g_min (rel to g0)", yscale="log")
    ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run_dir/"post_scale_diagnostics.png", dpi=160); plt.close(fig)
    # ------------------ end diagnostics ------------------





    # 6) exclusion
    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=det["target_snr"], g0=det["g0"], snr_efficiency=det["snr_eff"])




    # ========== INJECTION PRESENCE DIAGNOSTIC ==========


    # injection frequency (must exist when inj enabled)
    if inj["enabled"]:
        
        f_ax = ax_params.f_axion_hz if ax_params is not None else inj.get("f_axion_hz", None)
        if f_ax is None:
            print("[DIAG] No f_ax available.")
        else:
            # find nearest rebinned frequency index
            idx = np.argmin(np.abs(freqs_r - f_ax))
            half = len(Lq)//2
            win = 80  # rebinned bins to zoom around injection
            lo = max(0, idx - win)
            hi = min(len(freqs_r)-1, idx + win)

            print("[DIAG] injected f_ax (Hz)     :", f_ax)
            print("[DIAG] nearest freqs_r (Hz)   :", freqs_r[idx])
            print("[DIAG] grand Dg at idx        :", Dg[idx])
            if 'Dg_noinj' in globals():
                print("[DIAG] grand Dg_noinj at idx  :", Dg_noinj[idx])
                print("[DIAG] recovered (Dg-Dg_noinj) at idx:", (Dg - Dg_noinj)[idx])
            print("[DIAG] sg (noise sigma) at idx:", sg[idx])
            print("[DIAG] z-score at idx (Dg/sg):", Dg[idx] / sg[idx] if (np.isfinite(sg[idx]) and sg[idx]>0) else np.nan)
            print("[DIAG] matched-filter Rloc at idx:", Rloc[idx])
            print("[DIAG] gmin at idx (rel g0)   :", gmin[idx])

            # overlay plot: Dg, optional Dg_noinj, scaled template Lq centered at idx
            plt.figure(figsize=(8,4))
            plt.plot(freqs_r[lo:hi+1]/1e9, Dg[lo:hi+1], label="Dg (grand)")
            if 'Dg_noinj' in globals():
                plt.plot(freqs_r[lo:hi+1]/1e9, Dg_noinj[lo:hi+1], label="Dg_noinj", alpha=0.7)
            # template placed at idx-centered positions
            t_x = freqs_r[idx-half:idx-half+len(Lq)] / 1e9
            # offset template vertically to sit under Dg (use local median as baseline)
            baseline_level = np.nanmedian(Dg[lo:hi+1])
            plt.plot(t_x, baseline_level + Lq, marker='o', linestyle='None', label="Lq (scaled)", markersize=6, color="tab:red")
            plt.axvline(freqs_r[idx]/1e9, color="k", ls="--", label="injection nearest bin")
            plt.xlabel("Frequency [GHz]"); plt.ylabel("Grand-spectrum (arb)")
            plt.title("Zoom around injected frequency")
            plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir/"injection_zoom.png", dpi=160)
            plt.close()
    else:
        print("Injection not enabled; skip injection diagnostic.")
    # ========== end diagnostic ==========






    plot_exclusion(freqs_r, gmin, outfile=run_dir/"exclusion.png", title="95% CL Exclusion (SHM)")
    with (run_dir/"exclusion.csv").open("w") as fh:
        fh.write("freq_Hz,g_min_rel_to_g0\n")
        for f,g in zip(freqs_r, gmin):
            if np.isfinite(g): fh.write(f"{f},{g}\n")

    print(f"[OK] Run dir: {run_dir}")
    print(f"Candidates flagged: {len(cands)}  (threshold = {theta:.2f}σ)")


if __name__ == "__main__":
    main()