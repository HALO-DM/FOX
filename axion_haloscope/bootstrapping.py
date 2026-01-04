import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

def find_threshold_from_curve(powers, y_perc, threshold=95.0):
    """Find first crossing x where y_perc >= threshold using linear interpolation.
       Returns np.nan if no crossing exists.
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
        csv_path,
        n_boot=1000,
        threshold=95.0,
        default_n_trials=250,
        random_seed=1234,
    ):
    """
    csv_path: path to CSV with columns 'power','success' and optional 'n_trials'.
              'success' is integer number of successes at that power (0..n_trials)
    n_boot: number of bootstrap samples
    threshold: percentage threshold (95.0)
    default_n_trials: used when CSV has no 'n_trials' column
    Returns: dict with bootstrap thresholds array and summary stats
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
        vec = np.concatenate([np.ones(succ, dtype=np.uint8), np.zeros(ntr - succ, dtype=np.uint8)])
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
        boot_thresh[b] = find_threshold_from_curve(powers, y_perc, threshold=threshold)

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

# Example: run and plot


def plot_pretty_bootstrap_hist(
        thresholds,
        original_value=None,
        n_boot=None,
        n_valid=None,
        outfile="bootstrap_threshold_hist_pretty.png",
        bins=80,
        figsize=(8,5),
        title="Bootstrap distribution of 95% recovery threshold",
):
    """
    thresholds: 1D array-like of valid bootstrap thresholds (NaNs already removed)
    original_value: optional original point estimate to mark (e.g. 21.6)
    n_boot, n_valid: optional ints to annotate (n_valid <= n_boot)
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
    ylim = ax.get_ylim()
    ax.fill_betweenx([0, 3], p16, p84, color='C1', alpha=0.20, label='68% CI')
    ax.fill_betweenx([0, 3], p2p5, p97p5, color='C1', alpha=0.12, label='95% CI')

    # Plot histogram bars (lighter) and smooth curve (bold)
    ax.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.45, edgecolor='k', linewidth=0.4)
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
    stats = bootstrap_thresholds_from_csv("power.csv", n_boot=10000, default_n_trials=250)
    plot_pretty_bootstrap_hist(stats["all_thresholds"], original_value=21.6,
                           n_boot=1000, n_valid=stats["n_valid"],
                           outfile="bootstrap_threshold_hist_pretty.png")
    print("Bootstrap results (threshold where recovery >=95%):")
    print(f"  median = {stats['median']:.5f}")
    print(f"  mean   = {stats['mean']:.5f}")
    print(f"  std    = {stats['std']:.5f}")
    print(f"  68% CI ~ [{stats['p16']:.5f}, {stats['p84']:.5f}]")
    print(f"  95% CI ~ [{stats['p2p5']:.5f}, {stats['p97p5']:.5f}]")

