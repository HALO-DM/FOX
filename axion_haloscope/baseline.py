# axion_haloscope/baseline.py
from __future__ import annotations
from typing import Optional, Union, Dict, Tuple
import numpy as np
from scipy.signal import savgol_filter

def remove_baseline(
    spectrum: np.ndarray,
    window_length: int = 401,
    polyorder: int = 4,
    subtract_one: bool = False,
    diagnostic: Optional[Union[bool, Dict]] = None,
    freqs_hz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, "matplotlib.figure.Figure"]:
    """
    Savitzky–Golay baseline removal.

    Returns:
      (processed, baseline)            when diagnostic is falsy or diagnostic={"outfile": ...}
      (processed, baseline, figure)    when diagnostic is True or dict without 'outfile'
    """
    # --- compute baseline & processed
    baseline = savgol_filter(spectrum, window_length, 1, mode="interp")
    processed = spectrum / baseline
    if subtract_one:
        processed = processed - 1.0

    # --- optional diagnostics
    if diagnostic:
        import matplotlib.pyplot as plt
        cfg = {} if diagnostic is True else dict(diagnostic)
        x = freqs_hz if (freqs_hz is not None) else np.arange(len(spectrum))
        xlab = "Frequency [GHz]" if freqs_hz is not None else "Bin"
        if freqs_hz is not None:
            x = np.asarray(freqs_hz, float) / 1e9

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})

        # top: raw + baseline
        ax1.plot(x, spectrum, lw=0.6, label="raw")
        ax1.plot(x, baseline, lw=1.0, color="tab:red", label="baseline (SG)")
        ax1.set_ylabel("Power [arb]")
        ax1.set_title(cfg.get("title", "Baseline removal diagnostic"))
        ax1.grid(alpha=0.3); ax1.legend()

        # bottom: processed (green)
        ax2.plot(x, processed, lw=0.7, color="tab:green", label="processed")
        ax2.set_xlabel(xlab); ax2.set_ylabel("Processed")
        ax2.grid(alpha=0.3); ax2.legend()

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


# --- simple bin masking utility -------------------------------------------------
from typing import Iterable, Tuple, Optional

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
