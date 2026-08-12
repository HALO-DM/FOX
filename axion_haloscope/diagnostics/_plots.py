"""
Diagnostic Plots to plot results from axion_haloscope.diagnostics.evaluate_set_spacing
"""

__all__ = ['vary_set_size_plots']

from pathlib import Path
from typing import Optional, Dict, Tuple, List

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from axion_haloscope.baseline import remove_baseline
from axion_haloscope.utils import create_directory

def _compute_resonance_spread(
    var_set:List[Tuple[np.ndarray, np.ndarray, float]],
) -> Optional[float]:
    """Return the max-min spread of finite resonance frequencies in a set.

    Returns None if fewer than 2 finite resonance frequencies are available
    (either because `var_set` has fewer than 2 entries, or because too many
    entries have non-finite frequencies).
    """
    if len(var_set) < 2:
        return None
    res_freqs = np.asarray([item[2] for item in var_set], dtype=float)
    finite_vals = res_freqs[np.isfinite(res_freqs)]
    if len(finite_vals) < 2:
        return None
    return np.max(finite_vals) - np.min(finite_vals)


def _get_representative_set(
    sets_by_spacing: Dict[int, List[List[Tuple[np.ndarray, np.ndarray, float]]]],
    spacing: int,
) -> Optional[List[Tuple[np.ndarray, np.ndarray, float]]]:
    """Return the first non-empty set for a given spacing.
    
    Returns None if all sets at that spacing are empty (or `spacing` has no sets at all).
    """
    var_sets = sets_by_spacing[spacing]
    return next((s for s in var_sets if len(s) > 0), None)


def _plot_default_graph(
    x: np.ndarray,
    y: np.ndarray,
    save_dir: Path,
    x_label: str,
    y_label: str,
    title: str,
    file_name: str,
) -> None:
    """Basic graph template"""
    _, ax = plt.subplots(figsize=(13, 7))
    ax.plot(x, y, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{file_name}", dpi=150, bbox_inches='tight')
    plt.close()




def _plot_set_and_average_spectra(
    rep_set: List[Tuple[np.ndarray, np.ndarray, float]],
    freqs_avg: np.ndarray,
    spec_avg: np.ndarray,
    test_spacing: int,
    var_dir: Path
) -> None:
    """Plot all spectra in a set and the set average over the top"""
    fig, ax = plt.subplots(figsize=(13, 7))
    greys_cmap = mpl.colormaps['Greys']
    greys = greys_cmap(np.linspace(0.3, 0.9, len(rep_set)))
    for i_spec, x in enumerate(rep_set):
        ax.plot(x[1] / 1e6, x[0], color=greys[i_spec])
    ax.plot(freqs_avg, spec_avg, alpha=0.8, color="red", label="set averaged")
    norm = mcolors.Normalize(vmin=0, vmax=len(rep_set))
    sm = ScalarMappable(cmap=greys_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Spectrum index in set")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Set-averaged spectra and individual spectra — spacing {test_spacing} "
                 f"minutes (n={len(rep_set)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(var_dir / f"set_and_average_spectra_spacing_{test_spacing}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_sg_fit_overlay(
    spacings_config: List[int],
    sets_by_spacing: Dict[int, List[List[Tuple[np.ndarray, np.ndarray, float]]]],
    var_dir: Path,
    base: Dict[int, int, int, int, float, float, str, int],
) -> None:
    """Plot the SG fit of 1 set with all different spacing overlayed"""
    fig, ax = plt.subplots(figsize=(13, 7))
    viridis_cmap = mpl.colormaps['viridis']
    viridis = viridis_cmap(np.linspace(0, 1, len(spacings_config)))

    for c_idx, test_spacing in enumerate(spacings_config):
        rep_set = _get_representative_set(sets_by_spacing, test_spacing)
        if rep_set is None:
            continue

        freqs_avg = np.mean([x[1] for x in rep_set], axis=0)
        spec_avg = np.mean([x[0] for x in rep_set], axis=0)
        try:
            _, baseline = remove_baseline(
                spectrum=spec_avg,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
            )
        except ValueError as e:
            print(f"[SG overlay] spacing={test_spacing}minutes: SG fit failed ({e}), skipping")
            continue

        ax.plot(freqs_avg / 1e6, baseline, color=viridis[c_idx],
                label=f"{test_spacing} min  (n={len(rep_set)})")

    norm_spacing = mcolors.Normalize(vmin=min(spacings_config), vmax=max(spacings_config))
    sm_res = ScalarMappable(cmap=viridis_cmap, norm=norm_spacing)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=ax, label="Set size [minutes]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Set averaged spectra for set size variation (1st set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{var_dir}/sg_fit_set_size_var.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_average_spectra_with_errors(
    freqs_avg: np.ndarray,
    spec_avg: np.ndarray,
    spec_std: np.ndarray,
    test_spacing: int,
    n_spectra: int,
    var_dir: Path
) -> None:
    """Plot set average with error bars using standard deviation calculated on the residuals"""
    _, ax = plt.subplots(figsize=(13, 7))
    ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.10, ecolor="blue",
                color="white", label="std on average")
    ax.plot(freqs_avg, spec_avg, alpha=1, color="red", label="set averaged")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Set-averaged spectra with errors — spacing {test_spacing}"
                 f"minutes (n={n_spectra})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{var_dir}/average_spectra_errors_spacing_{test_spacing}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_average_spectra_with_errors_zoomed(
    freqs_avg: np.ndarray,
    spec_avg: np.ndarray,
    spec_std: np.ndarray,
    test_spacing: int,
    n_spectra: int,
    var_dir: Path,
    x_min: float=1.5,
    x_max: float=1.75
) -> None:
    """Zoomed in plot of set average with error bars using standard deviation calculated on 
    the residuals"""
    _, ax = plt.subplots(figsize=(13, 7))
    ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.5, ecolor="blue", color="white",
                label="std on average")
    ax.plot(freqs_avg, spec_avg, alpha=1, color="red", label="set averaged")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title(f"Set-averaged spectra with errors zoomed - spacing {test_spacing} "
                 f"minutes (n={n_spectra})")

    ax.set_xlim(x_min, x_max)
    in_range = (freqs_avg >= x_min) & (freqs_avg <= x_max)
    if in_range.any():
        y_lower = np.min(spec_avg[in_range] - spec_std[in_range])
        y_upper = np.max(spec_avg[in_range] + spec_std[in_range])
        y_pad = 0.05 * (y_upper - y_lower)
        ax.set_ylim(y_lower - y_pad, y_upper + y_pad)

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{var_dir}/average_spectra_errors_zoom_spacing_{test_spacing}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_avg_of_avg_std_vs_set_size(
    spacings_config: List[int],
    sets_by_spacing: Dict[int, List[List[Tuple[np.ndarray, np.ndarray, float]]]],
    var_dir: Path
) -> None:
    """Plot the average of all average set standard deviations (caluclated on the residuals)
    against the changing set size (minutes)"""
    spacing_avg_of_avg_std = []
    for spacing in spacings_config:
        var_sets = sets_by_spacing[spacing]
        set_avg_stds = [np.mean(np.std([x[0] for x in set], axis=0))
                        for set in var_sets if len(set) > 0]
        spacing_avg_of_avg_std.append(np.mean(set_avg_stds) if set_avg_stds else np.nan)

    _plot_default_graph(spacings_config, spacing_avg_of_avg_std, var_dir, "Set spacing [minutes]",
                             "Average standard deviation  [V²/Hz]",
                             "Average standard deviation vs set size", "avg_std_vs_set_size.png")


def _plot_resonance_drift_vs_set_size(
    spacings_config: List[int],
    sets_by_spacing: Dict[int, List[List[Tuple[np.ndarray, np.ndarray, float]]]],
    var_dir: Path
) -> None:
    """Average resonance of a set against changing set size"""
    spacing_avg_res_spread = []
    for spacing in spacings_config:
        var_sets = sets_by_spacing[spacing]
        res_spreads_this = [
            spread for var_set in var_sets
            if (spread := _compute_resonance_spread(var_set)) is not None
        ]
        spacing_avg_res_spread.append(np.mean(res_spreads_this) if res_spreads_this else np.nan)

    _plot_default_graph(spacings_config, spacing_avg_res_spread, var_dir, "Set size  [minutes]",
                        "Average resonance frequency spread  [GHz]",
                        "Cavity resonance drift within a set vs. set size",
                        "resonance_drift_vs_set_size.png")


def _plot_per_spacing_diagnostics(
    spacings_config: List[int],
    sets_by_spacing: Dict[int, List[List[Tuple[np.ndarray, np.ndarray, float]]]],
    var_dir: Path
) -> None:
    """Generate the set/average/zoomed diagnostic plots for each spacing."""
    for test_spacing in tqdm(spacings_config, desc="Set size variation diagnostic plots"):

        save_dir = create_directory(var_dir, f"spacing_{test_spacing}_minutes")

        rep_set = _get_representative_set(sets_by_spacing, test_spacing)
        if rep_set is None:
            continue

        freqs_avg = np.mean([x[1] for x in rep_set], axis=0) / 1e6
        spec_avg = np.mean([x[0] for x in rep_set], axis=0)
        spec_std = np.std([x[0] for x in rep_set], axis=0)

        _plot_set_and_average_spectra(rep_set, freqs_avg, spec_avg, test_spacing, save_dir)
        _plot_average_spectra_with_errors(freqs_avg, spec_avg, spec_std, test_spacing,
                                          len(rep_set), save_dir)
        _plot_average_spectra_with_errors_zoomed(freqs_avg, spec_avg, spec_std, test_spacing,
                                                 len(rep_set), save_dir)


def vary_set_size_plots(
    var_results: Dict[int, int, float, float, float, int, int, float],
    spacings_config: List[int],
    sets_by_spacing: Dict[int, List[List[tuple[np.ndarray, np.ndarray, float]]]],
    var_dir: str,
    base: Dict[int, int, int, int, float, float, str, int],
) -> None:
    """
    Generate the full suite of set-size-variation diagnostic plots.
    
    Parameters
    ----------
    var_results: Dict{
            spacing_minutes:  int, units: Minutes
                maximum time between the start and end of a set - the length of 
                time for which spectra are grouped into sets
            n_sets: int
                total number of sets 
            average_set_size: float
                average number of spectra in a set
            average_residual_std: float
                average standard deviation of the residuals of all sets
            average_residual_average: float
                average mean of the residuals of all sets
            total_masked: int
                total number of masked bins
            total_bins: int
                total number of bins
            fraction_masked: float
                total_masked / total_bins
            } 
        results/output of axion_haloscope.diagnostics.evaluate_set_spacing
    spacings_config: List[int]
        list of all spacing integers (e.g 30 minutes, 60 minutes etc)
    sets_by_spacing: Dict{sets}
        dictionary of grand sets each with different spacings according to spacings_config
    var_dir: str
        directory location to save plots generated with this function
    base: Dict{...}
        dictionary of all baseline settings from YAML
    
    See Also
    --------
    axion_haloscope.utils.load_yaml_config for full "base" description
    axion_haloscope.diagnostics.evaluate_set_spacing
    """
    spacings_plot = [r["spacing_minutes"] for r in var_results]
    resid_av      = [r["average_residual_average"] for r in var_results]
    resid_std     = [r["average_residual_std"] for r in var_results]
    total_masked  = [r["total_masked"] for r in var_results]

    _plot_default_graph(spacings_plot, resid_av, var_dir, "Set size [minutes]",
                        "Average residuals  [V²/Hz]",
                        f"Residual mean vs. set size ({base['clipping_mode']} clipping mode)",
                        "residual_avg_vs_set_size.png")
    _plot_default_graph(spacings_plot, resid_std, var_dir, "Set size threshold  [minutes]",
                        "Average residual std  [V²/Hz]",
                        f"Residual std vs. set size ({base['clipping_mode']} clipping mode)",
                        "residual_std_vs_set_size.png")
    _plot_sg_fit_overlay(spacings_config, sets_by_spacing, var_dir, base)
    _plot_per_spacing_diagnostics(spacings_config, sets_by_spacing, var_dir)
    _plot_avg_of_avg_std_vs_set_size(spacings_config, sets_by_spacing, var_dir)
    _plot_resonance_drift_vs_set_size(spacings_config, sets_by_spacing, var_dir)
    _plot_default_graph(spacings_config, total_masked, var_dir, "Set size [minutes]",
                        "Total masked bins ",
                        f"Total masked bins vs. set size ({base['clipping_mode']} clipping mode)",
                        "total_masked_vs_set_size.png")
