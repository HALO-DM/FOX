"""
step2_baselineremoval_warmhaystac.py

HAYSTAC-style warm baseline removal for haloscope test data.

Method
------
  0.  (Optional) Global gain alignment: shift each PSD in y so that all
      spectra share the same global-median mean PSD value.  Controlled by
      GAIN_ALIGN.  Offsets are saved to the output pickle.
  1.  Load step1_data.pkl
  2.  Group spectra into time-contiguous groups (each spanning at most
      GROUP_TIME_MINUTES of wall-clock time).
  3.  For each group: plot all cropped spectra overlaid using
      pltwCFfreqcol (one figure per group — visual check).
  4.  For each group: average all spectra onto a common frequency grid
      to form a single group-averaged spectrum.
  5.  Plot all group-averaged spectra overlaid (one figure).
  6.  Fit a Savitzky-Golay curve to each group-averaged spectrum.
  7.  Replot group-averaged spectra with SG fits overlaid (one figure).
  8.  Iterative sigma-clipping loop (N_ITER rounds):
        a. Mask bins deviating more than SIGMA_CUT * sigma from the fit.
        b. Refit SG on unmasked bins only (linear interp to fill gaps).
        c. Plot all groups on one figure: unmasked in group colour,
           masked bins in MASKED_COLOUR. Title says "Iteration <n>".
  9.  Divide every individual spectrum in each group by that group's
      final SG baseline → warm baseline removed.
  10. Trim edge bins to remove SG upturn artefacts — cuts EDGE_TRIM_FRAC
      of SG_WINDOW bins from each end of every baseline-removed spectrum.
  10. Save result to OUTPUT_PICKLE.
"""

import os
import pickle
import warnings
import pathlib
import yaml
import numpy as np
import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from AnalysisSubFuncs.PlotterFuncs import pltwCFfreqcol

# Set default config path
DEFAULT_CONFIG = "configs/step2_baselineremoval_warmhaystac.yaml"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman"],
    "font.size":   16,
})
# ===========================================================================
# LOAD CONFIG
# ===========================================================================

def _get(d, key, default):
    v = d.get(key, default)
    return default if v is None else v

def load_yaml_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    dat  = raw.get("data",            {}) or {}
    sgf  = raw.get("savitsky_golay",  {}) or {}
    clp   = raw.get("clipping",       {}) or {}
    aln = raw.get("alignment",        {}) or {}
    grp   = raw.get("graphing",       {}) or {}


    cfg = {
        "data": {
            "INPUT_PICKLE":       str(_get(dat, "INPUT_PICKLE", "step1b_data.pkl")),
            "OUTPUT_PICKLE":      str(_get(dat, "OUTPUT_PICKLE", "step2_data.pkl")),
            "PLOT_DIR":           str(_get(dat, "PLOT_DIR", "plots_step2")),
            "GROUP_TIME_MINUTES": int(_get(dat, "GROUP_TIME_MINUTES", 5.70e9)),
        },
        "savitsky_golay": {
            "SG_WINDOW":          int(_get(sgf, "SG_WINDOW", 351)),
            "SG_ORDER":           int(_get(sgf, "SG_ORDER", 4)),
        },
        "clipping": {
            "EDGE_TRIM_FRAC":     float(_get(clp, "EDGE_TRIM_FRAC", 0.25)),
            "SIGMA_CUT":          float(_get(clp, "SIGMA_CUT", 3.5)),
            "N_ITER":             int(_get(clp, "N_ITER", 3)),
        },
        "alignment": {
            "GAIN_ALIGN":         bool(_get(aln, "GAIN_ALIGN", True)),
        },
        "graphing": {
            "SAVE_FIGS":          bool(_get(grp, "SAVE_FIGS", True)),
            "SHOW_FIGS":          bool(_get(grp, "SHOW_FIGS", False)),
            "MASKED_COLOUR":      str(_get(grp, "MASKED_COLOUR", "crimson" )),
            "FIT_COLOUR":         str(_get(grp, "FIT_COLOUR", "gold")),
        },
    }
    return cfg

# ===========================================================================
# LOAD CONSTANTS
# ===========================================================================


#ap = argparse.ArgumentParser(description="Simulate haloscope run from YAML config")
#ap.add_argument("config", help="Path to YAML config (e.g. configs/simulate_run.yaml)")
#args = ap.parse_args()

#cfg_path = pathlib.Path(args.config).resolve()
#if not cfg_path.exists():
#    sys.exit(f"Config file not found: {cfg_path}")


ap = argparse.ArgumentParser(description="Simulate haloscope run from YAML config")
#ap.add_argument("config", help="Path to YAML config (e.g. configs/simulate_run.yaml)")
ap.add_argument("config", nargs='?', help="Path to YAML config (uses default if not provided)")
args = ap.parse_args()

# Use default if no config provided
if not args.config:
    cfg_path = pathlib.Path(DEFAULT_CONFIG).resolve()
    print(f"Using default config: {cfg_path}")
else:
    cfg_path = pathlib.Path(args.config).resolve()

if not cfg_path.exists():
    sys.exit(f"Config file not found: {cfg_path}")

# Load the configuration
cfg = load_yaml_config(cfg_path)

# Unpack the configuration
dat, sgf, clp, aln, grp = (cfg[k] for k in ("data","savitsky_golay","clipping","alignment","graphing"))

# ===========================================================================
# USER PARAMETERS
# ===========================================================================
INPUT_PICKLE  = dat["INPUT_PICKLE"]
OUTPUT_PICKLE = dat["OUTPUT_PICKLE"]
PLOT_DIR      = dat["PLOT_DIR"]

# --- Grouping ---
GROUP_TIME_MINUTES = dat["GROUP_TIME_MINUTES"]      # max span of a single group [min]


# --- Savitzky-Golay ---
SG_WINDOW = sgf["SG_WINDOW"]                # must be odd
SG_ORDER  = sgf["SG_ORDER"]

# --- Iterative sigma clipping ---
SIGMA_CUT = clp["SIGMA_CUT"]                  # bins further than this * sigma are masked
N_ITER    = clp["N_ITER"]                    # number of clipping iterations

# --- Global gain alignment (applied before grouping) ---
GAIN_ALIGN  = aln["GAIN_ALIGN"]         # toggle global gain alignment on/off
                                 # Each PSD receives a y-shift so that all PSDs
                                 # share the same global-median mean PSD value.

# --- Edge trimming (Stage 10) ---
# Fraction of SG_WINDOW to cut from each side of every baseline-removed
# spectrum.  E.g. 0.15 with SG_WINDOW=1051 removes ~157 bins either side.
# Set to 0.0 to disable.
EDGE_TRIM_FRAC = clp["EDGE_TRIM_FRAC"]

# --- Plotting ---
SAVE_FIGS     = grp["SAVE_FIGS"]           # save PNGs to PLOT_DIR
SHOW_FIGS     = grp["SHOW_FIGS"]            # plt.show() after every figure (set True for interactive)
MASKED_COLOUR = grp["MASKED_COLOUR"]        # colour for masked bins in iteration plots
FIT_COLOUR    = grp["FIT_COLOUR"]


# ===========================================================================

os.makedirs(PLOT_DIR, exist_ok=True)
matplotlib.use("Agg")  # non-interactive; change to "TkAgg" etc. if SHOW_FIGS=True


def _savefig(fig, name):
    if SAVE_FIGS:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {path}")
    if SHOW_FIGS:
        plt.show(block=False)
    plt.close(fig)


def _sg_reflect(y, window, order):
    """SG fit with reflected-edge padding to suppress edge artefacts."""
    if window >= len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
        window = max(order + 2, window)
        window = window + 1 if window % 2 == 0 else window
    if window < order + 2:
        return np.full_like(y, np.nanmean(y))
    pad    = window // 2
    padded = np.pad(y, pad_width=pad, mode="reflect")
    smooth = savgol_filter(padded, window, order)
    return smooth[pad: pad + len(y)]


def _sg_masked(freqs, psd, mask_bad, window, order):
    """
    Refit SG using only unmasked bins, then interpolate back onto full grid.
    """
    good = ~mask_bad
    if good.sum() < window:
        return _sg_reflect(psd, window, order)
    fit_good = _sg_reflect(psd[good], window, order)
    fit_full = np.interp(freqs, freqs[good], fit_good)
    return fit_full

# ===========================================================================
# LOAD DATA
# ===========================================================================

print("\n" + "=" * 60)
print("Loading step1 data ...")
print("=" * 60)

with open(INPUT_PICKLE, "rb") as fh:
    data = pickle.load(fh)

spectra   = data["Power_Spectra"]
res_freqs = np.asarray(data["res_freq"],  dtype=float)
cw_freqs  = np.asarray(data["cw_freqs"],  dtype=float)
filenames = data["filenames"]
N         = len(filenames)

_raw_times = data.get("attr::Date-Time", [None] * N)
timestamps = []
for raw in _raw_times:
    try:
        timestamps.append(datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        timestamps.append(None)

print(f"  {N} spectra loaded, "
      f"{sum(1 for t in timestamps if t is not None)} with valid timestamps")

# ===========================================================================
# STAGE 0 — Global gain alignment  (optional, pre-grouping)
# ===========================================================================

gain_align_offsets = np.zeros(N, dtype=float)   # zero if GAIN_ALIGN=False

if GAIN_ALIGN:
    print("\n" + "=" * 60)
    print("Stage 0: Global gain alignment  (aligning all PSD means)")
    print("=" * 60)

    psd_means = np.full(N, np.nan)
    for i in range(N):
        s = spectra[i]
        if s is None:
            continue
        p_i = np.asarray(s[1], dtype=float)
        if p_i.size > 0:
            psd_means[i] = np.nanmean(p_i)
    finite_mask = np.isfinite(psd_means)
    if not np.any(finite_mask):
        print("  WARNING: no finite PSD means found — gain alignment skipped")
    else:
        global_target = np.nanmedian(psd_means[finite_mask])
        print(f"  Global target mean  : {global_target:.6g}")
        print(f"  Mean spread (std)   : "
              f"{np.nanstd(psd_means[finite_mask]):.6g}  (before alignment)")

        n_corrected = 0
        for i in range(N):
            if spectra[i] is None or not np.isfinite(psd_means[i]):
                continue
            offset = global_target - psd_means[i]
            gain_align_offsets[i] = offset
            f_i = np.asarray(spectra[i][0], dtype=float)
            p_i = np.asarray(spectra[i][1], dtype=float)
            spectra[i] = [f_i, p_i + offset]
            data["Power_Spectra"][i] = spectra[i]
            n_corrected += 1

        aligned_means = psd_means[finite_mask] + gain_align_offsets[finite_mask]
        print(f"  Mean spread (std)   : "
              f"{np.nanstd(aligned_means):.6g}  (after alignment)")
        print(f"  Corrected {n_corrected} / {N} spectra")

else:
    print("\nStage 0: Global gain alignment disabled — skipping")


# ===========================================================================
# STAGE 1 — Group spectra by wall-clock time
# ===========================================================================

print("\n" + "=" * 60)
print(f"Stage 1: Grouping spectra  (max {GROUP_TIME_MINUTES} min per group)")
print("=" * 60)

group_indices = []
current_group = []
group_start_t = None

for i in range(N):
    if spectra[i] is None:
        continue
    t = timestamps[i]
    if t is None:
        if not current_group:
            group_start_t = None
        current_group.append(i)
        continue
    if group_start_t is None:
        group_start_t = t
        current_group.append(i)
    else:
        elapsed_min = (t - group_start_t).total_seconds() / 60.0
        if elapsed_min <= GROUP_TIME_MINUTES:
            current_group.append(i)
        else:
            if current_group:
                group_indices.append(current_group)
            current_group = [i]
            group_start_t = t

if current_group:
    group_indices.append(current_group)

n_groups = len(group_indices)
print(f"  => {n_groups} groups formed")
for g, idx_list in enumerate(group_indices):
    t0 = timestamps[idx_list[0]]
    t1 = timestamps[idx_list[-1]]
    print(f"     Group {g:3d}: {len(idx_list):4d} spectra  "
          f"| {str(t0)} -> {str(t1)}")


# ===========================================================================
# STAGE 2 — Per-group overlay plot
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 2: Per-group cropped spectra overlay plots")
print("=" * 60)

_cmap_det = plt.cm.inferno

for g, idx_list in enumerate(group_indices):
    print(f"  Plotting Group {g} ({len(idx_list)} spectra) ...")

    _sp  = [spectra[i]  for i in idx_list]
    _cw  = cw_freqs[idx_list]
    _res = res_freqs[idx_list]
    _cv  = np.abs(_cw - _res) * 1e9
    _fin = _cv[np.isfinite(_cv)]
    _norm_g = Normalize(
        vmin=np.percentile(_fin,  0) if len(_fin) else 0,
        vmax=np.percentile(_fin, 100) if len(_fin) else 1,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for s, cv in zip(_sp, _cv):
        if s is None or not np.isfinite(cv):
            continue
        ax.plot(
            np.asarray(s[0]) * 1e-6,
            np.asarray(s[1]),
            linestyle="", marker="o", markersize=2,
            color=_cmap_det(_norm_g(cv)), alpha=0.55,
        )
    sm = ScalarMappable(cmap=_cmap_det, norm=_norm_g)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$|f_{\rm CW} - f_{\rm res}|$  [Hz]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Group {g} — cropped spectra overlay  ({len(idx_list)} spectra)")
    plt.tight_layout()
    _savefig(fig, f"group{g:03d}_A_overlay.png")


# ===========================================================================
# STAGE 3 — Form group-averaged spectra
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 3: Forming group-averaged spectra")
print("=" * 60)

group_avg_spectra = []

for g, idx_list in enumerate(group_indices):
    valid = [
        (np.asarray(spectra[i][0], dtype=float),
         np.asarray(spectra[i][1], dtype=float))
        for i in idx_list if spectra[i] is not None
    ]
    if not valid:
        group_avg_spectra.append(None)
        print(f"  Group {g}: no valid spectra — skipped")
        continue

    f_min  = max(f.min() for f, _ in valid)
    f_max  = min(f.max() for f, _ in valid)
    n_bins = int(np.median([len(f) for f, _ in valid]))

    if f_min >= f_max:
        f_grid = valid[0][0]
    else:
        f_grid = np.linspace(f_min, f_max, n_bins)

    stack = []
    for f, p in valid:
        ip = interp1d(f, p, bounds_error=False, fill_value=np.nan)
        stack.append(ip(f_grid))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psd_avg = np.nanmean(np.array(stack), axis=0)

    group_avg_spectra.append((f_grid, psd_avg))
    print(f"  Group {g}: {len(valid)} spectra averaged, "
          f"{n_bins} bins, "
          f"f=[{f_min*1e-6:.3f}, {f_max*1e-6:.3f}] MHz")


# ===========================================================================
# STAGE 4 — Plot all group-averaged spectra (no fits yet)
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 4: Plotting all group-averaged spectra")
print("=" * 60)

_cmap_g = plt.cm.viridis
_group_mean_res = np.array([
    np.nanmean(res_freqs[idx_list]) for idx_list in group_indices
])

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

fig, ax = plt.subplots(figsize=(13, 5))
for g, avg in enumerate(group_avg_spectra):
    if avg is None:
        continue
    f, p = avg
    ax.plot(f * 1e-6, p, lw=1.2, alpha=0.8, color=_gcol(g), label=f"Grp {g}")
sm_res = ScalarMappable(cmap=_cmap_g, norm=_norm_res)
sm_res.set_array([])
fig.colorbar(sm_res, ax=ax, label="Mean cavity resonance  [GHz]")
ax.set_xlabel("IF frequency  [MHz]")
ax.set_ylabel("PSD  [V²/Hz]")
ax.set_title("Group-averaged spectra — all groups")
ax.set_xlim(0,2)
plt.tight_layout()
_savefig(fig, "all_groups_B_averaged.png")
print(_group_mean_res)

# ===========================================================================
# STAGE 5 — First SG fit to each group-averaged spectrum + replot with fits
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 5: Initial SG fits")
print("=" * 60)

group_sg_fits = []
for g, avg in enumerate(group_avg_spectra):
    if avg is None:
        group_sg_fits.append(None)
        continue
    f, p = avg
    fit  = _sg_reflect(p, SG_WINDOW, SG_ORDER)
    group_sg_fits.append(fit)
    print(f"  Group {g}: fit done")

fig, ax = plt.subplots(figsize=(13, 5))
for g, (avg, fit) in enumerate(zip(group_avg_spectra, group_sg_fits)):
    if avg is None or fit is None:
        continue
    f, p = avg
    col  = _gcol(g)
    ax.plot(f * 1e-6, p,   lw=1.0, alpha=0.55, color=col, label=f"Grp {g}")
    ax.plot(f * 1e-6, fit, lw=1.8, alpha=0.95, color=col, linestyle="--")
sm_res2 = ScalarMappable(cmap=_cmap_g, norm=_norm_res)
sm_res2.set_array([])
fig.colorbar(sm_res2, ax=ax, label="Mean cavity resonance  [GHz]")
ax.set_xlabel("IF frequency  [MHz]")
ax.set_ylabel("PSD  [V²/Hz]")
ax.set_title("Group-averaged spectra with initial SG fits  (dashed = fit)")
plt.tight_layout()
_savefig(fig, "all_groups_C_with_initial_fit.png")


# ===========================================================================
# STAGE 6 — Iterative sigma-clipping
# ===========================================================================

print("\n" + "=" * 60)
print(f"Stage 6: Iterative sigma-clipping  "
      f"({N_ITER} iterations, sigma_cut={SIGMA_CUT})")
print("=" * 60)

group_masks = [
    np.zeros(len(avg[0]), dtype=bool) if avg is not None else None
    for avg in group_avg_spectra
]

for iteration in range(1, N_ITER + 1):
    print(f"\n  --- Iteration {iteration} / {N_ITER} ---")
    total_new = 0

    for g, avg in enumerate(group_avg_spectra):
        if avg is None:
            continue
        f, p         = avg
        current_mask = group_masks[g]
        prev_fit     = group_sg_fits[g]
        if prev_fit is None:
            continue

        residuals = p - prev_fit
        sigma     = np.std(residuals[~current_mask])
        new_bad   = np.abs(residuals) > SIGMA_CUT * sigma
        combined  = current_mask | new_bad
        n_new     = int(np.sum(new_bad & ~current_mask))
        total_new += n_new

        new_fit = _sg_masked(f, p, combined, SG_WINDOW, SG_ORDER)

        group_masks[g]   = combined
        group_sg_fits[g] = new_fit

        print(f"    Group {g:3d}: sigma={sigma:.4g}  "
              f"newly masked={n_new:4d}  "
              f"total masked={int(combined.sum()):4d}/{len(f)}")

    print(f"  Total newly masked this iteration: {total_new}")

    fig, ax = plt.subplots(figsize=(14, 5))
    for g, avg in enumerate(group_avg_spectra):
        if avg is None:
            continue
        f, p  = avg
        mask  = group_masks[g]
        fit   = group_sg_fits[g]
        col   = _gcol(g)

        ax.scatter(f[~mask] * 1e-6, p[~mask],
                   s=4,  color=col,          alpha=0.6)
        ax.scatter(f[ mask] * 1e-6, p[ mask],
                   s=12, color=MASKED_COLOUR, alpha=0.85, zorder=5)
        if fit is not None:
            ax.plot(f * 1e-6, fit, lw=1.6, color=col, linestyle="--", alpha=0.9)

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor="grey",       markersize=7,  label="Unmasked bins"),
            plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=MASKED_COLOUR, markersize=9, label="Masked bins"),
            plt.Line2D([0], [0], color="grey", ls="--", lw=1.5,    label="SG fit"),
        ]
        ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
        sm_iter = ScalarMappable(cmap=_cmap_g, norm=_norm_res)
        sm_iter.set_array([])
        fig.colorbar(sm_iter, ax=ax, label="Mean cavity resonance  [GHz]")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Iteration {iteration}  —  masked bins in {MASKED_COLOUR}  "
                    f"(sigma_cut={SIGMA_CUT}, window={SG_WINDOW})")
        plt.tight_layout()
        _savefig(fig, f"all_groups_D_iteration{iteration:02d}_group{g}.png")
        sys.exit()

# ===========================================================================
# STAGE 7 — Apply group baselines to individual spectra (divide out)
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 7: Applying group SG baselines to individual spectra")
print("=" * 60)

n_applied = 0
n_skipped = 0

for g, idx_list in enumerate(group_indices):
    avg = group_avg_spectra[g]
    fit = group_sg_fits[g]

    if avg is None or fit is None:
        print(f"  Group {g}: no baseline — skipping {len(idx_list)} spectra")
        n_skipped += len(idx_list)
        continue

    f_grid          = avg[0]
    baseline_interp = interp1d(
        f_grid, fit,
        bounds_error=False,
        fill_value=(fit[0], fit[-1]),
    )

    for i in idx_list:
        s = spectra[i]
        if s is None:
            n_skipped += 1
            continue
        f_i   = np.asarray(s[0], dtype=float)
        psd_i = np.asarray(s[1], dtype=float)

        bl      = baseline_interp(f_i)
        bl_safe = np.where(np.abs(bl) > 1e-40, bl, np.nanmean(psd_i))
        data["Power_Spectra"][i] = [f_i, psd_i / bl_safe]
        n_applied += 1

print(f"  Applied: {n_applied}  |  Skipped: {n_skipped}")

# ===========================================================================
# STAGE 8 — Post-removal summary plots
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 8: Post-removal summary plots")
print("=" * 60)

print("  Figure E1: all baseline-removed spectra (pltwCFfreqcol) ...")
pltwCFfreqcol(data, "Post warm-baseline removal — all spectra (PSD / group baseline)")
fig_e1 = plt.gcf()
fig_e1.axes[0].axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
plt.tight_layout()
_savefig(fig_e1, "all_groups_E1_post_removal_allspectra.png")

print("  Figure E2: group-averaged residuals ...")
fig2, ax2 = plt.subplots(figsize=(13, 5))
for g, avg in enumerate(group_avg_spectra):
    if avg is None:
        continue
    f_g, p_g = avg
    fit_g    = group_sg_fits[g]
    if fit_g is None:
        continue
    bl_safe = np.where(np.abs(fit_g) > 1e-40, fit_g, np.nanmean(p_g))
    ratio   = p_g / bl_safe - 1.0
    ax2.plot(f_g * 1e-6, ratio, lw=1.0, color=_gcol(g), alpha=0.7)
ax2.axhline(0.0, color="k", ls="--", lw=0.8)
sm_e2 = ScalarMappable(cmap=_cmap_g, norm=_norm_res)
sm_e2.set_array([])
fig2.colorbar(sm_e2, ax=ax2, label="Mean cavity resonance  [GHz]")
ax2.set_xlabel("IF frequency  [MHz]")
ax2.set_ylabel("(avg − fit) / fit")
ax2.set_title("Group-averaged residuals after final SG fit  (should be ~0)")
plt.tight_layout()
_savefig(fig2, "all_groups_E2_group_residuals.png")

# ===========================================================================
# STAGE 9 — Per-group baseline-removed overlay plots
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 9: Subtracting final SG baseline and plotting per-group overlays")
print("=" * 60)

for g, idx_list in enumerate(group_indices):
    avg = group_avg_spectra[g]
    fit = group_sg_fits[g]

    if avg is None or fit is None:
        print(f"  Group {g}: no baseline — skipping")
        continue

    f_grid          = avg[0]
    baseline_interp = interp1d(
        f_grid, fit,
        bounds_error=False,
        fill_value=(fit[0], fit[-1]),
    )

    print(f"  Group {g}: subtracting baseline from {len(idx_list)} spectra ...")

    sub_spectra = []
    for i in idx_list:
        s = data["Power_Spectra"][i]
        if s is None:
            sub_spectra.append(None)
            continue
        f_i   = np.asarray(s[0], dtype=float)
        psd_i = np.asarray(s[1], dtype=float)
        sub_spectra.append([f_i, psd_i])

    _cw  = cw_freqs[idx_list]
    _res = res_freqs[idx_list]
    _cv  = np.abs(_cw - _res) * 1e9
    _fin = _cv[np.isfinite(_cv)]
    _norm_g = Normalize(
        vmin=np.percentile(_fin,  0) if len(_fin) else 0,
        vmax=np.percentile(_fin, 100) if len(_fin) else 1,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for s, cv in zip(sub_spectra, _cv):
        if s is None or not np.isfinite(cv):
            continue
        ax.plot(
            np.asarray(s[0]) * 1e-6,
            np.asarray(s[1]),
            linestyle="", marker="o", markersize=2,
            color=_cmap_det(_norm_g(cv)), alpha=0.55,
        )
    sm = ScalarMappable(cmap=_cmap_det, norm=_norm_g)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$|f_{\rm CW} - f_{\rm res}|$  [Hz]")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("(PSD / baseline) − 1")
    ax.set_title(f"Group {g} — baseline-subtracted spectra overlay  "
                 f"({len(idx_list)} spectra)")
    plt.tight_layout()
    _savefig(fig, f"group{g:03d}_F_baseline_removed.png")

print(f"  Per-group baseline-removed overlay plots saved to '{PLOT_DIR}/'")


# ===========================================================================
# STAGE 10 — Edge-bin trimming to remove SG upturn artefacts
# ===========================================================================
#
# The SG fit can turn up at the edges of the frequency window due to the
# reflected padding in _sg_reflect.  After dividing by the SG baseline,
# those edge bins end up biased away from 1.  This stage trims n_trim bins
# from each end of every baseline-removed spectrum, where:
#
#   n_trim = round(EDGE_TRIM_FRAC * SG_WINDOW)
#
# Typical values: EDGE_TRIM_FRAC = 0.10–0.20.  Increase if the upturn
# extends further inward.  Set EDGE_TRIM_FRAC = 0.0 to disable entirely.
# ===========================================================================

n_trim = int(np.round(EDGE_TRIM_FRAC * SG_WINDOW))

print("\n" + "=" * 60)
print(f"Stage 10: Edge-bin trimming  "
      f"(EDGE_TRIM_FRAC={EDGE_TRIM_FRAC}, SG_WINDOW={SG_WINDOW})")
print(f"  Trimming {n_trim} bins from each edge of every spectrum")
print("=" * 60)

if n_trim <= 0:
    print("  n_trim = 0 — no trimming applied")
else:
    n_trimmed      = 0
    n_trim_skipped = 0

    for i in range(N):
        s = data["Power_Spectra"][i]
        if s is None:
            n_trim_skipped += 1
            continue

        f_i   = np.asarray(s[0], dtype=float)
        psd_i = np.asarray(s[1], dtype=float)

        if len(f_i) <= 2 * n_trim:
            # Spectrum too short to trim — leave untouched
            n_trim_skipped += 1
            continue

        data["Power_Spectra"][i] = [f_i[n_trim:-n_trim],
                                    psd_i[n_trim:-n_trim]]
        n_trimmed += 1

    print(f"  Trimmed : {n_trimmed}")
    print(f"  Skipped : {n_trim_skipped}")

    # ── Post-trim overlay plot ────────────────────────────────────────────────
    _cv_trim   = np.abs(cw_freqs - res_freqs) * 1e9
    _fin_trim  = _cv_trim[np.isfinite(_cv_trim)]
    _norm_trim = Normalize(
        vmin=np.percentile(_fin_trim,  0) if len(_fin_trim) else 0,
        vmax=np.percentile(_fin_trim, 100) if len(_fin_trim) else 1,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, s in enumerate(data["Power_Spectra"]):
        if s is None or not np.isfinite(_cv_trim[i]):
            continue
        ax.plot(
            np.asarray(s[0]) * 1e-6,
            np.asarray(s[1]),
            linestyle="", marker="o", markersize=2,
            color=_cmap_det(_norm_trim(_cv_trim[i])), alpha=0.55,
        )
    sm_trim = ScalarMappable(cmap=_cmap_det, norm=_norm_trim)
    sm_trim.set_array([])
    fig.colorbar(sm_trim, ax=ax,
                 label=r"$|f_{\rm CW} - f_{\rm res}|$  [Hz]")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD / baseline")
    ax.set_title(
        f"Stage 10 — post edge-trim overlay  "
        f"({n_trim} bins removed each side,  "
        f"frac={EDGE_TRIM_FRAC}, window={SG_WINDOW})"
    )
    plt.tight_layout()
    _savefig(fig, "all_groups_G_post_edge_trim.png")

    data["edge_trim_n_bins"] = n_trim
    data["edge_trim_frac"]   = EDGE_TRIM_FRAC


# ===========================================================================
# SAVE
# ===========================================================================

data["warm_baseline_groups"]      = group_indices
data["warm_baseline_avg_spectra"] = group_avg_spectra
data["warm_baseline_sg_fits"]     = group_sg_fits
data["warm_baseline_masks"]       = group_masks
data["gain_align_offsets"]        = gain_align_offsets

with open(OUTPUT_PICKLE, "wb") as fh:
    pickle.dump(data, fh)

print("\n" + "=" * 60)
print("Step 2 (warm HAYSTAC-style baseline removal) complete")
print(f"  Global gain alignment   : {'ON' if GAIN_ALIGN else 'OFF'}")
print(f"  Groups              : {n_groups}")
print(f"  Group time window   : {GROUP_TIME_MINUTES} min")
print(f"  SG window / order   : {SG_WINDOW} / {SG_ORDER}")
print(f"  Sigma cut / iters   : {SIGMA_CUT} / {N_ITER}")
print(f"  Edge trim           : {n_trim} bins each side  "
      f"(frac={EDGE_TRIM_FRAC})")
print(f"  Spectra processed   : {n_applied}  (skipped: {n_skipped})")
print(f"  Output pickle       : {OUTPUT_PICKLE}")
print(f"  Plots saved to      : {PLOT_DIR}/")
print("=" * 60)
