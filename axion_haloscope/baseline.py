# axion_haloscope/baseline.py
"""
Baseline removal
========

Baseline removal, spectrum alignment, and bin masking utilities for axion haloscope spectra.

This module provides the core pre-processing steps used before downstream analysis: removing
slowly-varying baseline structure from a spectrum (Savitzky-Golay fit), aligning and averaging
multiple spectra that may sit on different or irregularly-sampled x-axes, and masking out
known-bad bins (e.g. RFI) by index or frequency range.

Functions
---------
remove_baseline
    Savitzky-Golay baseline removal from a single spectrum, with
    optional diagnostic plotting.
align_and_average_spectra
    Align multiple (x, y) spectra onto a common x-axis and compute
    their NaN-aware average.
mask_bins
    Mask spectrum bins by explicit index or by frequency range.
"""
from __future__ import annotations
from typing import Optional, Union, Dict, Tuple, List, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def remove_baseline(
    spectrum: np.ndarray,
    window_length: int = 401,
    polyorder: int = 4,
    mode: str = "multiplicative",
    subtract_one: Optional[bool] = False,
    add_one: Optional[bool] = None,
    diagnostic: Optional[Union[bool, Dict]] = None,
    freqs_hz: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, "matplotlib.figure.Figure"]:
    """
    Savitzky-Golay baseline removal. Switch included to determine which removal method to use.
    Diagnostic graph also can be optionally outputted. One of core plots.

    Parameters
    ----------
    spectrum : 1D ndarray
        Spectrum to remove the baseline from.
    window_length : int, optional
        Length of the SG filter window. Default is 401.
    polyorder : int, optional
        Order of the polynomial used in the SG fit. Default is 4.
    mode : {"multiplicative", "additive"}, optional
        Baseline removal mode. Default is "multiplicative".
    subtract_one : bool, optional
        If True, subtracts 1 from the residuals after baseline removal (shifts down by 1).
        Default is False.
    add_one : bool, optional
        If True, adds 1 to the residuals after baseline removal (shifts up by 1). Default is
        None no shift).
    diagnostic : bool or dict, optional
        If True, also return a diagnostic figure. If a dict, controls diagnostic behavior via
        optional keys:

        - "title" : str, custom plot title
        - "outfile" : str, path to save the figure to disk; if given, the figure is saved and
          closed rather than returned
        - "show" : bool, if True the figure is displayed rather than returned (default False)

        Default is None (no diagnostic).
    freqs_hz : ndarray, optional
        Frequency axis in Hz, used for diagnostic plot x-axis labeling. If omitted, bin index
        is used instead.
    baseline : ndarray, optional
        Precomputed baseline to use instead of recomputing via SG.

    Returns
    -------
    processed : ndarray
        Residuals from baseline removal.
    baseline : ndarray
        Baseline produced from the SG algorithm.
    fig : matplotlib.figure.Figure, optional
        Diagnostic plot showing the spectrum and baseline, plus the residuals below. 
        Only returned when `diagnostic` is True, or a dict with neither "outfile" nor "show" set.

    Raises
    ------
    ValueError
        If `mode` is not one of "additive" or "multiplicative".

    Notes
    -----
    Runs `savgol_filter` internally (mode="interp") to compute the baseline, unless `baseline`
    is supplied directly.

    Return arity depends on `diagnostic`:

        (processed, baseline)        diagnostic falsy, or dict with "outfile" or "show"=True
        (processed, baseline, fig)   diagnostic is True, or dict without "outfile"/"show"

    Caution: when `diagnostic` is a dict with "show": True and no
    "outfile", the figure is currently neither returned nor closed —
    it stays open. Callers using this option should call `plt.close()`
    on their own figures if this matters.

    See Also
    --------
    scipy.signal.savgol_filter : for more details on the actual smoothing algorithm
    """
    # --- compute baseline & processed
    if baseline is None:
        baseline = savgol_filter(spectrum, window_length, polyorder, mode="interp")
    if mode == "additive":
        processed = spectrum - baseline
    elif mode == "multiplicative":
        processed = spectrum / baseline
    else:
        raise ValueError("Please select either Additive or Multiplicative Baseline Removal")
    if subtract_one:
        processed = processed - 1.0
    if add_one:
        processed = processed + 1.0

    # --- optional diagnostics
    if diagnostic:
        cfg = {} if diagnostic is True else dict(diagnostic)
        x = freqs_hz if (freqs_hz is not None) else np.arange(len(spectrum))
        xlab = "Frequency [GHz]" if freqs_hz is not None else "Bin"
        if freqs_hz is not None:
            x = np.asarray(freqs_hz, float)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})


        ax1.plot(x, spectrum, lw=0.6, label="raw")
        ax1.plot(x, baseline, lw=1.0, color="tab:red", label="baseline (SG)")
        ax1.set_ylabel("Power [arb]")
        ax1.set_title(cfg.get("title", "Baseline removal diagnostic"))
        ax1.grid(alpha=0.3)
        ax1.legend()

        # bottom: processed (green)
        ax2.plot(x, processed, lw=0.7, color="tab:green", label="processed")
        ax2.set_xlabel(xlab)
        ax2.set_ylabel("Processed")
        ax2.grid(alpha=0.3)
        ax2.legend()

        fig.tight_layout()

        # save-to-file path → return two-tuple (no figure kept)
        if cfg.get("outfile"):
            fig.savefig(cfg["outfile"], dpi=150)
            plt.close(fig)
            return processed, baseline

        # no outfile → return the figure as well (caller decides to save/close)
        if not cfg.get("show", False):
            return processed, baseline, fig
        # if show=True, fall through and return two-tuple below (figure left open)

    # ALWAYS return a tuple
    return processed, baseline

def align_and_average_spectra(
    xs_list: List[np.ndarray],
    ys_list: List[np.ndarray],
    round_decimals: Optional[int] = None,
    preserve_first_seen: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Align a list of (x, y) spectra onto a common x-axis and return
    (common_x, padded_y, average_y, baseline_averages).

    Each spectrum's x-values are merged into a single sorted (or
    first-seen-ordered) common axis. Each spectrum's y-values are then
    scattered into a common-length array at the positions matching their
    x-values, with NaN filling any positions a given spectrum doesn't
    cover. The average is taken across spectra, ignoring NaNs.

    Parameters
    ----------
    xs_list : list of 1D arrays
        List of 1D x-axis arrays, one per spectrum. Must be the same
        length as `ys_list`, with matching shapes element-wise.
    ys_list : list of 1D arrays
        List of 1D y-axis arrays, one per spectrum, paired with `xs_list`.
    round_decimals : int, optional
        If given, round all x-values to this many decimal places before
        aligning. Useful when spectra should share x-values but differ by
        floating-point noise. Default is None (no rounding).
    preserve_first_seen : bool, optional
        If True, order `common_x` by the order in which each unique x-value
        was first encountered (scanning `xs_list` in order), rather than
        numerically sorting it. Default is False (numerically sorted, via
        `np.unique`'s default behavior).

    Returns
    -------
    common_x : 1D array
        1D array of the union of all x-values across `xs_list`, either
        sorted (default) or ordered by first appearance (if
        `preserve_first_seen` is True).
    padded : 2D array of shape (n_spectra, len(common_x))
        Row `i` holds spectrum `i`'s y-values placed at the columns matching their
        x-position in `common_x`, with NaN elsewhere.
    average : 1D array
        1D array of length `len(common_x)`, the NaN-ignoring mean of
        `padded` across spectra (columns with no data from any spectrum
        are NaN).
    baseline_averages : list of 1D arrays
        One array per input spectrum, same length and x-order as the
        corresponding entry in `xs_list`, giving `average` evaluated at
        that spectrum's original x positions.

    Raises
    ------
    ValueError
        If `xs_list` and `ys_list` have different lengths, or if any
        corresponding pair `xs_list[i]`, `ys_list[i]` have mismatched
        shapes.

    Notes
    -----
    Duplicate x-values within a single spectrum are not summed or
    averaged against each other — each occurrence maps to its own
    position via `inv`, so `padded` can have multiple rows contributing
    to the same column index only across *different* spectra, not within
    one.

    A manual NaN-mean fallback is included for environments where
    `np.nanmean` is unavailable; columns with zero valid entries are
    explicitly set to NaN to avoid a divide-by-zero warning surfacing
    as anything other than NaN.
    """
    xs_list = [np.asarray(x) for x in xs_list]
    ys_list = [np.asarray(y) for y in ys_list]

    if len(xs_list) != len(ys_list):
        raise ValueError("xs_list and ys_list must have the same length")

    for i, (x, y) in enumerate(zip(xs_list, ys_list)):
        if x.shape != y.shape:
            raise ValueError(f"xs_list[{i}] and ys_list[{i}] must have same shape")

    if round_decimals is not None:
        xs_list = [np.round(x, round_decimals) for x in xs_list]

    concatenated = np.concatenate(xs_list)
    if preserve_first_seen:
        uniq_vals, first_idx = np.unique(concatenated, return_index=True)
        order = np.argsort(first_idx)
        common_x = uniq_vals[order]
        mapper = {val: i for i, val in enumerate(common_x)}
        inv = np.fromiter((mapper[v] for v in concatenated), dtype=int, count=concatenated.size)
    else:
        common_x, inv = np.unique(concatenated, return_inverse=True)

    n_spectra = len(xs_list)
    m = common_x.size
    padded = np.full((n_spectra, m), np.nan, dtype=float)

    # keep per-spectrum index arrays so we can build baseline_averages later
    idxs: List[np.ndarray] = []
    pos = 0
    for i, y in enumerate(ys_list):
        n = xs_list[i].size
        idx = inv[pos:pos + n]           # indices into common_x for this spectrum
        idxs.append(idx.copy())
        padded[i, idx] = y.astype(float)
        pos += n

  # average across spectra ignoring NaNs; fallback if numpy lacks nanmean
    try:
        average = np.nanmean(padded, axis=0)
    except AttributeError:
        valid = ~np.isnan(padded)
        counts = valid.sum(axis=0)
        sums = np.nansum(padded, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            average = sums / counts
        average[counts == 0] = np.nan

    # build baseline averages per spectrum (preserve original order & duplicates)
    baseline_averages = [average[idx] for idx in idxs]

    return common_x, padded, average, baseline_averages

# --- simple bin masking utility -------------------------------------------------
def mask_bins(
    spectrum: np.ndarray,
    *,
    freqs_hz: Optional[np.ndarray] = None,
    indices: Optional[Iterable[int]] = None,
    ranges_hz: Optional[Iterable[Tuple[float, float]]] = None,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mask bins in a spectrum by index or by frequency ranges.

    Parameters
    ----------
    spectrum : 1D array
        Input spectrum to be masked (not modified in place).
    freqs_hz : 1D array, optional
        Frequencies for each bin [Hz]. Required if using ranges_hz.
    indices : iterable of int, optional
        Exact bin indices to mask (e.g., known RFI bins).
    ranges_hz : iterable of (fmin, fmax), optional
        Frequency ranges [Hz] to mask. Requires freqs_hz.
    fill_value : float
        Value to write into masked bins (default: NaN).

    Returns
    -------
    masked : 1D array
        Copy of spectrum with masked bins set to `fill_value`.
    mask : 1D bool array
        True where a bin is masked.
    """
    x = np.asarray(spectrum, float)
    m = np.zeros_like(x, dtype=bool)

    # mask by explicit indices
    if indices is not None:
        idx = np.asarray(list(indices), dtype=int)
        idx = idx[(idx >= 0) & (idx < x.size)]
        m[idx] = True

    # mask by frequency ranges
    if ranges_hz is not None:
        if freqs_hz is None:
            raise ValueError("freqs_hz is required when masking by ranges_hz.")
        f = np.asarray(freqs_hz, float)
        for fmin, fmax in ranges_hz:
            if fmin > fmax:
                fmin, fmax = fmax, fmin
            m |= (f >= fmin) & (f <= fmax)

    y = x.copy()
    y[m] = fill_value
    return y, m
