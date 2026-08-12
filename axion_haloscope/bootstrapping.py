"""
Please note: Code is Potentially Redundant - used exclusivly for a few plots for Blue's 
1st Semester Report, no functions are used currently in analysis
"""
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _find_threshold_from_curve(
    powers: np.ndarray,
    y_perc: np.ndarray,
    threshold: float=95.0,
) -> float:
    """
    Find the first x-value where a curve crosses a given threshold.

    Scans `y_perc` for the first index where it is at or above `threshold`, then linearly
    interpolates between that point and the preceding one to estimate the crossing location
    in `powers`.

    Parameters
    ----------
    powers : 1D array
        x-values corresponding to `y_perc`, assumed sorted ascending.
    y_perc : 1D array
        y-values (e.g. cumulative percentage) to search for a threshold
        crossing, same length as `powers`.
    threshold : float
        Value that `y_perc` must reach or exceed. Default is 95.0.

    Returns
    -------
    crossing : float
        Interpolated x-value in `powers` where `y_perc` first reaches `threshold`. If the very
        first point already meets the threshold, that point's x-value is returned directly (no
        interpolation, since there is no preceding point). Returns `np.nan` if `y_perc` never 
        reaches `threshold`.

    Notes
    -----
    If the two points bracketing the crossing have equal y-values (`y1 == y0`), interpolation
    would divide by zero; in that case the later point's x-value is returned directly instead.
    """
    powers = np.asarray(powers, dtype=float)
    y = np.asarray(y_perc, dtype=float)
    idx = np.where(y >= threshold)[0]
    if idx.size == 0:
        return np.nan
    i1 = int(idx[0])
    if i1 == 0:
        return float(powers[0])
    x0, y0 = float(powers[i1-1]), float(y[i1-1])
    x1, y1 = float(powers[i1]), float(y[i1])
    if y1 == y0:
        return float(x1)
    t = (threshold - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)

def bootstrap_thresholds_from_csv(
    csv_path: Path,
    n_boot: int=1000,
    threshold: float=95.0,
    default_n_trials: int=250,
    random_seed: int=1234,
) -> Dict:
    """
    Bootstrap the threshold crossing point from binomial trial data in a CSV.

    Reads per-power success counts from a CSV, reconstructs the underlying 0/1 trial outcomes
    at each power, and repeatedly resamples (with replacement) to build a distribution of
    threshold crossing points via `_find_threshold_from_curve`. Summary statistics over the
    bootstrap distribution are returned.

    Parameters
    ----------
    csv_path : Path
        Path to a CSV file with columns "power" and "success", and optionally "n_trials".
        "success" is the integer number of successes observed at that power (between 0 and
        `n_trials` inclusive).
    n_boot : int
        Number of bootstrap resamples to draw. Default is 1000.
    threshold : float
        Percentage threshold passed to `_find_threshold_from_curve` for each bootstrap sample.
        Default is 95.0.
    default_n_trials : int
        Number of trials to assume per row when the CSV has no "n_trials" column. Default is 250.
    random_seed : int
        Seed for the random number generator, for reproducibility. Default is 1234.

    Returns
    -------
    stats : dict
        Summary of the bootstrap distribution, with keys:
        - "n_boot" : int, number of bootstrap samples requested.
        - "n_valid" : int, number of samples that produced a valid threshold crossing
        (i.e. not NaN).
        - "median" : float, median of the valid bootstrap thresholds.
        - "mean" : float, mean of the valid bootstrap thresholds.
        - "std" : float, sample standard deviation (ddof=1) of the valid bootstrap thresholds.
        - "p16", "p84" : float, 16th and 84th percentiles (~1-sigma band for a normal 
        distribution).
        - "p2p5", "p97p5" : float, 2.5th and 97.5th percentiles (~95% interval).
        - "all_thresholds" : ndarray, the valid bootstrap threshold values themselves (NaN
        entries removed).

    Raises
    ------
    ValueError
        If the CSV is missing a "power" or "success" column, or if any row's "success" value is
        negative or exceeds that row's "n_trials".
    RuntimeError
        If every bootstrap sample fails to produce a threshold crossing (all NaN), typically
        meaning the power range doesn't reach `threshold` or `threshold` is set too high.

    Notes
    -----
    Each CSV row is expanded into an explicit vector of `n_trials` zeros and ones
    (success/failure), which is then resampled with replacement per bootstrap
    iteration — this is a standard nonparametric bootstrap over binomial trial data rather
    than a parametric (e.g. beta-binomial) resampling.

    Monotonicity of the resampled percentage curve is not enforced by default (an optional
    `np.maximum.accumulate` step is present but commented out in the implementation), so
    `_find_threshold_from_curve` may pick up a spurious early crossing if the underlying data
    is noisy and non-monotonic.

    See Also
    --------
    _find_threshold_from_curve : threshold crossing search used per bootstrap sample.
    """
    df = pd.read_csv(csv_path)
    if "power" not in df.columns or "success" not in df.columns:
        raise ValueError("CSV must contain 'power' and 'success' columns.")
    powers = df["power"].values
    # per-row total trials
    if "n_trials" in df.columns:
        n_trials_arr = df["n_trials"].astype(int).values
    else:
        n_trials_arr = np.full(len(df), int(default_n_trials))

    # reconstruct per-power trial arrays (list of 0/1 arrays)
    trial_vectors = []
    for succ, ntr in zip(df["success"].astype(int).values, n_trials_arr):
        if succ < 0 or succ > ntr:
            raise ValueError("success must be between 0 and n_trials")
        vec = np.concatenate([np.ones(succ, dtype=np.uint8),
                              np.zeros(ntr - succ, dtype=np.uint8)])
        trial_vectors.append(vec)

    rng = np.random.default_rng(random_seed)
    boot_thresh = np.empty(n_boot, dtype=float)
    boot_thresh.fill(np.nan)

    for b in range(n_boot):
        y_perc = []
        for vec in trial_vectors:
            ntr = vec.size
            # resample with replacement
            sample = rng.choice(vec, size=ntr, replace=True)
            frac = sample.mean() * 100.0  # percent
            y_perc.append(frac)
        # (optionally enforce monotonicity: makes interpolation stable)
        # y_perc = np.maximum.accumulate(y_perc)   # uncomment to force non-decreasing
        boot_thresh[b] = _find_threshold_from_curve(powers, y_perc, threshold=threshold)

    # drop nan (cases where no crossing occurred in a bootstrap sample)
    valid = ~np.isnan(boot_thresh)
    n_valid = valid.sum()
    if n_valid == 0:
        raise RuntimeError("No bootstrap sample produced a crossing >= threshold. "
                           "Consider increasing power range or lower threshold.")

    boot_valid = boot_thresh[valid]

    stats = {
        "n_boot": n_boot,
        "n_valid": int(n_valid),
        "median": float(np.median(boot_valid)),
        "mean": float(np.mean(boot_valid)),
        "std": float(np.std(boot_valid, ddof=1)),
        "p16": float(np.percentile(boot_valid, 16)),
        "p84": float(np.percentile(boot_valid, 84)),
        "p2p5": float(np.percentile(boot_valid, 2.5)),
        "p97p5": float(np.percentile(boot_valid, 97.5)),
        "all_thresholds": boot_valid,  # numpy array
    }
    return stats

def plot_pretty_bootstrap_hist(
    thresholds: np.ndarray,
    outfile: str="bootstrap_threshold_hist_pretty.png",
    bins: int=80,
    figsize: Tuple=(8,5),
    title: str ="Bootstrap distribution of 95% recovery threshold",
) -> Dict:
    """
    Plot a styled histogram of bootstrap threshold values and save to file.

    Draws a density histogram of `thresholds` with a smoothed (Gaussian-convolved) density curve
    overlaid, shaded 68% and 95% confidence bands, and vertical lines marking the median and mean.
    The figure is saved to `outfile` and closed; summary statistics are returned to the caller.

    Parameters
    ----------
    thresholds : 1D array
        Array-like of valid bootstrap threshold values (NaNs should already be removed, e.g. via
        the "all_thresholds" entry from `bootstrap_thresholds_from_csv`).
    outfile : str, optional
        Path to save the figure to. Default is "bootstrap_threshold_hist_pretty.png".
    bins : int, optional
        Number of histogram bins. Default is 80.
    figsize : tuple, optional
        Figure size in inches, passed to `plt.figure`. Default is (8, 5).
    title : str, optional
        Plot title. Default is "Bootstrap distribution of 95% recovery threshold".

    Returns
    -------
    stats : dict
        Summary statistics of `thresholds`, with keys:
        - "median" : float, median value.
        - "mean" : float, mean value.
        - "std" : float, sample standard deviation (ddof=1); 0.0 if `thresholds` has fewer
        than 2 elements.
        - "p16", "p84" : float, 16th and 84th percentiles (68% CI bounds, shaded on the plot).
        - "p2p5", "p97p5" : float, 2.5th and 97.5th percentiles (95% CI bounds, shaded on the
        plot).
        - "n_valid" : int, number of values in `thresholds`.

    Raises
    ------
    ValueError
        If `thresholds` is empty.

    """
    th = np.asarray(thresholds, dtype=float)
    if th.size == 0:
        raise ValueError("No valid thresholds to plot (empty array).")

    # summary stats
    med = np.median(th)
    mean = np.mean(th)
    std = np.std(th, ddof=1) if th.size > 1 else 0.0
    p16, p84 = np.percentile(th, [16, 84])
    p2p5, p97p5 = np.percentile(th, [2.5, 97.5])

    # set up figure with a small inset axis for boxplot underneath
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 0.6], hspace=0.12)
    ax = fig.add_subplot(gs[0])

    # Histogram (density)
    counts, bin_edges = np.histogram(th, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Smooth the histogram to make a KDE-like curve (simple Gaussian smoothing)
    # smoothing width in number of bins; adapt to data spread
    sigma_bins = max(1.0, (bins * 0.02))  # small default smoothing
    # build Gaussian kernel
    kernel_radius = int(3 * sigma_bins)
    kx = np.arange(-kernel_radius, kernel_radius+1)
    kernel = np.exp(-0.5 * (kx / sigma_bins)**2)
    kernel /= kernel.sum()
    smooth_density = np.convolve(counts, kernel, mode="same")

    # Shaded percentile bands
    ax.fill_betweenx([0, 3], p16, p84, color='C1', alpha=0.20, label='68% CI')
    ax.fill_betweenx([0, 3], p2p5, p97p5, color='C1', alpha=0.12, label='95% CI')

    # Plot histogram bars (lighter) and smooth curve (bold)
    ax.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.45, edgecolor='k',
            linewidth=0.4)
    ax.plot(bin_centers, smooth_density, lw=2.0, label="Smoothed density")


    # Vertical lines: median, mean, original value
    ax.axvline(med, color='C3', linestyle='--', lw=1.8, label=f"median = {med:.4g}")
    ax.axvline(mean, color='C4', linestyle=':', lw=1.4, label=f"mean = {mean:.4g}")

    ax.set_title(title)
    ax.set_ylabel("Density")
    ax.set_xlabel("Recovered threshold (power units)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.25)


    # trim whitespace
    plt.ylim(0, 1.75)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return {
        "median": med,
        "mean": mean,
        "std": std,
        "p16": p16, "p84": p84,
        "p2p5": p2p5, "p97p5": p97p5,
        "n_valid": th.size
    }

if __name__ == "__main__":
    bootstrap_stats = bootstrap_thresholds_from_csv("power.csv", n_boot=10000, default_n_trials=250)
    plot_pretty_bootstrap_hist(bootstrap_stats["all_thresholds"],
                               outfile="bootstrap_threshold_hist_pretty.png")
    print("Bootstrap results (threshold where recovery >=95%):")
    print(f"  median = {bootstrap_stats['median']:.5f}")
    print(f"  mean   = {bootstrap_stats['mean']:.5f}")
    print(f"  std    = {bootstrap_stats['std']:.5f}")
    print(f"  68% CI ~ [{bootstrap_stats['p16']:.5f}, {bootstrap_stats['p84']:.5f}]")
    print(f"  95% CI ~ [{bootstrap_stats['p2p5']:.5f}, {bootstrap_stats['p97p5']:.5f}]")
