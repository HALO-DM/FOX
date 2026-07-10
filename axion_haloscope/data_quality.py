# axion_haloscope/data_quality.py
from __future__ import annotations
from typing import Callable, Iterable, List, Tuple
import numpy as np
from .io import SpectrumSet

BadPredicate = Callable[[np.ndarray, np.ndarray, int], bool]

def placeholder_bad_predicate(s: np.ndarray, f: np.ndarray, i: int) -> bool:
    return False


def power_too_high(
    s: np.ndarray,
    f: np.ndarray,
    i: int,
    *,
    p_max: float = 1e-8,
) -> bool:
    """
    Flag a spectrum as BAD if its max power exceeds the max power limit p_max.
    - units are in the spectrum’s native (arb) units.
    """
    max_power = np.nanmax(s)

    return max_power > p_max

def spectra_is_zeros(
    s: np.ndarray,
    f: np.ndarray,
    i: int,
) -> bool:
    """
    Flag a spectrum as BAD if its power spectra is an array of zeros.
    """

    return np.all(s == 0)

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
        metadata=sset.metadata
    )
    removed = SpectrumSet(
        spectra=[sset.spectra[i] for i in bad],
        freqs_per_spec=[sset.freqs_per_spec[i] for i in bad],
        rf_grid=sset.rf_grid,
        rf_index_map=[sset.rf_index_map[i] for i in bad],
        metadata=sset.metadata
    )
    return filtered, removed, keep, bad



def restrict_frequency_range(
    sset,
    *,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
):
    """
    Keep only bins within [fmin_hz, fmax_hz] for each spectrum.

    This modifies the spectra, frequency axes, and rf_index_map consistently.
    The global rf_grid is also trimmed to the same range.
    """
    import numpy as np
    from .io import SpectrumSet

    if fmin_hz is None:
        fmin_hz = -np.inf
    if fmax_hz is None:
        fmax_hz = np.inf

    spectra_new = []
    freqs_new = []
    old_maps_new = []

    for s, f, idx in zip(sset.spectra, sset.freqs_per_spec, sset.rf_index_map):
        s = np.asarray(s)
        f = np.asarray(f)
        idx = np.asarray(idx)

        keep = (f >= fmin_hz) & (f <= fmax_hz)

        if np.any(keep):
            spectra_new.append(s[keep])
            freqs_new.append(f[keep])
            old_maps_new.append(idx[keep])

    if len(spectra_new) == 0:
        raise ValueError(
            f"No spectral bins left after frequency cut: "
            f"{fmin_hz} <= f <= {fmax_hz}"
        )

    # Build a new compact RF grid from the kept frequencies
    # and remap each spectrum onto it.
    all_freqs = np.concatenate(freqs_new)
    rf_grid_new = np.unique(all_freqs)

    mapper = {float(f): i for i, f in enumerate(rf_grid_new)}
    rf_index_map_new = [
        np.asarray([mapper[float(ff)] for ff in f], dtype=int)
        for f in freqs_new
    ]

    return SpectrumSet(
        spectra=spectra_new,
        freqs_per_spec=freqs_new,
        rf_grid=rf_grid_new,
        rf_index_map=rf_index_map_new,
    )
