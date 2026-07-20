# axion_haloscope/data_quality.py
from __future__ import annotations
from typing import Callable, Iterable, List, Tuple
import numpy as np
from .io import SpectrumSet

BadPredicate = Callable[[np.ndarray, np.ndarray, int], bool]

def placeholder_bad_predicate(s: np.ndarray, f: np.ndarray, i: int) -> bool:
    return False

def too_noisy(
    s: np.ndarray,
    f: np.ndarray,
    i: int,
    *,
    rms_max: float = 3.0,
    nan_fail: bool = True,
    robust: bool = True,
) -> bool:
    """
    Flag a spectrum as BAD if its (robust) RMS exceeds rms_max or contains NaNs/inf.
    - robust=True uses median+MAD; False uses mean+std.
    - units are in the spectrum’s native (arb) units.
    """
    if nan_fail and (not np.isfinite(s).all()):
        return True
    x = s
    if robust:
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))  # ≈ 0.6745 σ for Gaussian
        sigma = mad / 0.6744897501960817 if mad > 0 else np.nanstd(x)
        rms = np.sqrt(np.nanmean((x - med) ** 2))
    else:
        mu = np.nanmean(x)
        sigma = np.nanstd(x)
        rms = np.sqrt(np.nanmean((x - mu) ** 2))
    if not np.isfinite(sigma):  # degenerate edge case
        return True
    return rms > rms_max

def identify_bad_spectra(sset: SpectrumSet, predicate: BadPredicate | None = None) -> List[int]:
    pred = predicate or placeholder_bad_predicate
    bad: List[int] = []
    for i, (s, f) in enumerate(zip(sset.spectra, sset.freqs_per_spec)):
        try:
            if pred(s, f, i):
                bad.append(i)
        except Exception:
            bad.append(i)
    return bad

def filter_spectrum_set(
    sset: SpectrumSet,
    bad_indices: Iterable[int] | None = None,
    bad_mask: Iterable[bool] | None = None,
    predicate: BadPredicate | None = None,
) -> Tuple[SpectrumSet, List[int], List[int]]:
    n = sset.n_spectra()
    if sum(x is not None for x in (bad_indices, bad_mask, predicate)) > 1:
        raise ValueError("Provide only one of bad_indices, bad_mask, or predicate.")
    if bad_indices is not None:
        bad = sorted(set(int(i) for i in bad_indices if 0 <= int(i) < n))
    elif bad_mask is not None:
        m = list(bool(b) for b in bad_mask)
        if len(m) != n:
            raise ValueError(f"bad_mask length {len(m)} != n_spectra {n}")
        bad = [i for i, b in enumerate(m) if b]
    else:
        bad = identify_bad_spectra(sset, predicate=predicate)  # defaults to keep-all
    keep = [i for i in range(n) if i not in set(bad)]
    filtered = SpectrumSet(
        spectra=[sset.spectra[i] for i in keep],
        freqs_per_spec=[sset.freqs_per_spec[i] for i in keep],
        rf_grid=sset.rf_grid,
        rf_index_map=[sset.rf_index_map[i] for i in keep],
    )
    return filtered, keep, bad
