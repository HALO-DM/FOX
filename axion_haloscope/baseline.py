# axion_haloscope/baseline.py
from __future__ import annotations
from typing import Optional, Union, Dict, Tuple, List
import numpy as np
import matplotlib
from scipy.signal import savgol_filter

def remove_baseline(
    spectrum: np.ndarray,
    window_length: int = 401,
    polyorder: int = 4,
    mode: str = "multiplicative",
    subtract_one: bool = False,
    diagnostic: Optional[Union[bool, Dict]] = None,
    freqs_hz: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, "matplotlib.figure.Figure"]:
    """
    Savitzky–Golay baseline removal.

    Returns:
      (processed, baseline)            when diagnostic is falsy or diagnostic={"outfile": ...}
      (processed, baseline, figure)    when diagnostic is True or dict without 'outfile'
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

    # --- optional diagnostics
    if diagnostic:
        import matplotlib.pyplot as plt
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

def align_and_average_spectra(
  xs_list: List[np.ndarray],
  ys_list: List[np.ndarray],
  round_decimals: Optional[int] = None,
  preserve_first_seen: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
  """
  Align a list of (x, y) spectra onto a common x-axis and return
  (common_x, padded_y, average_y, baseline_averages).

  baseline_averages is a list of arrays: baseline_averages[i] has the same
  length/order as xs_list[i] and contains the global average evaluated at
  those x positions.
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
