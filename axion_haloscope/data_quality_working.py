# axion_haloscope/data_quality.py
from __future__ import annotations
from typing import Callable, Iterable, List, Tuple
import numpy as np
from datetime import datetime
from .io_working import SpectrumSet, SpectrumMetadata

import sys

BadPredicate = Callable[[np.ndarray, np.ndarray, SpectrumMetadata, int], bool]

def placeholder_bad_predicate(s: np.ndarray, f: np.ndarray, md: SpectrumMetadata, i: int) -> bool:
    return False



def power_too_high(
    s: np.ndarray,
    f: np.ndarray,
    md: SpectrumMetadata,
    i: int,
    *,
    p_max: float = 1e-8,
) -> bool:
    """
    Flag a spectrum as bad if its mean power exceeds the max power limit p_max.
    
    Parameters
    ----------
        s : np.ndarray
            power data
        f : np.ndarray
            frequency data
        md : SpectrumMetadata
            metadata for the spectrum.
        i : int
            spectrum index
        p_max : float
            max power threshold in arb units.
    
    Returns
    -------
        bool
            
        """
    max_power = np.max(s)

    return max_power > p_max



def small_bandwidth(
    s: np.ndarray,
    f: np.ndarray,
    md: SpectrumMetadata,
    i: int,
    *,
    bw_min: float = 0.00027,
) -> bool:
    """
    Flag a spectrum as bad if its minimium bandwidth is below the threshold.

    Parameters
    ----------
        s : np.ndarray
            power data
        f : np.ndarray
            frequency data
        md : SpectrumMetadata
            metadata for the spectrum.
        i : int
            spectrum index
        bw_min : float
            min bandwidth value, in Hz.

    Returns
    -------
        bool
        
    """
    bw = md.bandwidths[i]

    return bw < bw_min



def metadata_is_zeros(
    s: np.ndarray,
    f: np.ndarray,
    md: SpectrumMetadata,
    i: int,
    *,
    item: str,
) -> bool:
    """
    Flag a spectrum as bad if an attribute (specified by item) is full of zeros.

    Parameters
    ----------
        s : np.ndarray
            power data
        f : np.ndarray
            frequency data
        md : SpectrumMetadata
            metadata for the spectrum.
        i : int
            spectrum index
        item : str
            name of the metadata element that is being checked.

    Returns
    -------
        bool
        
    """
    value = getattr(md, item)
    return bool(np.all(value[i] == 0))



def time_filter(
    s: np.ndarray,
    f: np.ndarray,
    md: SpectrumMetadata,
    i: int,
    *,
    start_time: str,
    end_time: str, 
) -> bool:
    """
    Flag a spectrum as bad if it is within known bad time range.
    
    Parameters
    ----------
        s : np.ndarray
            power data
        f : np.ndarray
            frequency data
        md : SpectrumMetadata
            metadata for the spectrum.
        i : int
            spectrum index
        start_time : str,
            start time of the data that is being removed.
        end_time : str,
            end time of the data that is being removed.
    Returns
    -------
        bool
        
    """
    date = md.dates[i]

    if isinstance(date, str):
        date = datetime.fromisoformat(date)

    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)

    if start < date < end:
        return True
    else:
        return False



def too_noisy(
    s: np.ndarray,
    f: np.ndarray,
    md: SpectrumMetadata,
    i: int,
    *,
    rms_max: float = 3.0,
    nan_fail: bool = True,
    robust: bool = True,
) -> bool:
    """
    Flag a spectrum as bad if its (robust) RMS exceeds rms_max or contains NaNs/inf.
    - robust = True uses median+MAD; False uses mean+std
    - units are in the spectrum’s native (arb) units.

    Parameters
    ----------
        s : np.ndarray
            power data
        f : np.ndarray
            frequency data
        md : SpectrumMetadata
            metadata for the spectrum.
        i : int
            spectrum index
        rms_max : float
            maxmium value of rms.
        nan_fail : bool
            if checking for NaNs/inf
        robust : bool
            if using median+MAD (True) or mean+std (False)

    Returns
    -------
        bool
        
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



def identify_bad_spectra(sset: SpectrumSet,
                        predicate: BadPredicate | None = None
                        ) -> List[int]:
    """
    Finds the bad spectra considering a bad predicate and outputs a list of
    indices of the bad spectra. 

    Parameters
    ----------
        sset : SpectrumSet
            the spectra
        predicate : BadPredicate
            the condition that is being considered when filtering
    Returns
    -------
        List[int]
            list of the indices of the bad spectra

        
    """
    pred = predicate or placeholder_bad_predicate
    bad: List[int] = []
    for i, (s, f) in enumerate(zip(sset.spectra, sset.freqs_per_spec)):
        try:
            if pred(s, f, sset.metadata, i):
                bad.append(i)
        except Exception:
            bad.append(i)
    return bad

def _index_metadata(metadata: SpectrumMetadata, idx: List[int]) -> SpectrumMetadata:
    fields = vars(metadata)
    new_fields = {
        key: (
            value if key == "invalid_files"
            else (value[idx] if isinstance(value, np.ndarray) else [value[i] for i in idx])
        )
        for key, value in fields.items()
    }
    return SpectrumMetadata(**new_fields)

def filter_spectrum_set(
    sset: SpectrumSet,
    bad_indices: Iterable[int] | None = None,
    bad_mask: Iterable[bool] | None = None,
    predicate: BadPredicate | None = None,
) -> Tuple[SpectrumSet, SpectrumSet, List[int], List[int]]:
    
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
        metadata=_index_metadata(sset.metadata, keep),
    )
    removed = SpectrumSet(
        spectra=[sset.spectra[i] for i in bad],
        freqs_per_spec=[sset.freqs_per_spec[i] for i in bad],
        rf_grid=sset.rf_grid,
        rf_index_map=[sset.rf_index_map[i] for i in bad],
        metadata=_index_metadata(sset.metadata, bad),
    )
    return filtered, removed, keep, bad



def restrict_frequency_range(
    sset: SpectrumSet,
    *,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
):
    """
    Keep only bins within [fmin_hz, fmax_hz] for each spectrum.
    
    This modifies the spectra, frequency axes, and rf_index_map consistently.
    The global rf_grid is also trimmed to the same range.

    Parameters
    ----------
        sset : SpectrumSet
            the spectra
        fmin_hz : float
            lower limit on the accepted frequency range.
        fmax_hz : float
            upper limit on the accepted frequency range.

    Returns
    -------
        bool
        
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
