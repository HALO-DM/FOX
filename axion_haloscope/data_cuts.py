"""
data_cuts
=========

Performs cuts on the data, either w.r.t time or frequency

"""
from datetime import datetime

import numpy as np

from axion_haloscope.io_working import SpectrumSet, SpectrumMetadata

def cut_by_datetime(data, start, end):
    """
    Filter a SpectrumSet down to spectra taken within a datetime range.

    Parses `data.metadata.dates` and keeps only the spectra whose timestamp falls within
    `[start, end]` inclusive. Excluded spectra are logged into the returned metadata's
    `invalid_files` list rather than silently dropped.

    Parameters
    ----------
    data : SpectrumSet
        Input spectrum set, with `spectra`, `freqs_per_spec`, `rf_grid`, `rf_index_map`,
        and `metadata` attributes.
    start : str
        Start of the datetime range to keep, inclusive.
    end : str
        End of the datetime range to keep, inclusive.

    Returns
    -------
    SpectrumSet
        A new `SpectrumSet` containing only the spectra (and matching `freqs_per_spec`,
        `rf_index_map`, and per-spectrum metadata fields) whose date falls within
        `[start, end]`. `rf_grid` is carried over unchanged. The metadata's `invalid_files`
        is extended with an entry for each spectrum dropped by this call, each of the form 
        `[file_name, "not in good time range", date]`.

    """

    specs, fper, rf, rf_map, metadata = (data.spectra, data.freqs_per_spec, data.rf_grid,
                                         data.rf_index_map, data.metadata)

    dt = np.array([
        datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S") if x is not None else None
        for x in metadata.dates
    ])

    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d %H:%M:%S")

    mask = np.array([(d is not None) and (start_dt <= d <= end_dt) for d in dt])

    spectra        = [b for a, b in zip(mask, specs) if a]
    freqs_per_spec = [b for a, b in zip(mask, fper) if a]
    rf_index_map   = [b for a, b in zip(mask, rf_map) if a]

    removed = [[metadata.file_names[i], "not in good time range", metadata.dates[i]]
        for i, keep in enumerate(mask) if not keep]

    invalid = list(metadata.invalid_files)
    invalid_all = invalid + removed

    fields = vars(metadata)
    new_fields = {
        k: (
            invalid_all if k == "invalid_files"
            else (v[mask] if isinstance(v, np.ndarray) else [val for keep,
                                                             val in zip(mask, v) if keep])
        )
        for k, v in fields.items()
    }
    spec_metadata = SpectrumMetadata(**new_fields)

    return SpectrumSet(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        rf_grid=rf,
        rf_index_map=rf_index_map,
        metadata=spec_metadata
    )

def cut_by_values(
    sset: SpectrumSet,
    cut_min_val: float=-0.3e6,
    cut_max_val: float=2.3e6,
) -> SpectrumSet:
    """
    Trim each spectrum in a SpectrumSet to a fixed frequency window,
    after patching over any zero-frequency (DC) bins.

    Only to be 

    For each spectrum, any bin whose frequency is exactly 0 (e.g. a DC spike or LO leakage
    artifact) is patched by copying values from its immediate neighbors, then the spectrum,
    its frequency axis, and its RF index map are all truncated to the same index range.

    Parameters
    ----------
    sset : SpectrumSet
        Input spectrum set, with `spectra`, `freqs_per_spec`, `rf_grid`,
        `rf_index_map`, and `metadata` attributes.
    cut_min_val : float
        Lower bound of the frequency window to keep. 
    cut_max_val : float
        Upper bound of the frequency window to keep.

    Returns
    -------
    SpectrumSet
        A new `SpectrumSet` with each spectrum's data, frequency axis, and RF index map truncated
        to the cut window, and `rf_grid` truncated to match. `metadata` is carried over unchanged.

    Notes
    -----
    Cut indices are found once, from spectrum 0's frequency axis (`fper[0]`), via nearest-value
    lookup, and then applied identically to every spectrum in the set — this assumes all spectra
    share the same frequency axis.

    The DC-bin patch replaces each zero-frequency bin and its two neighbors on either side 
    (5 bins total per occurrence: the found index plus 2 bins on each side) with the value 
    immediately outside that window, propagating inward.
    """

    specs, fper, rf, rf_map, metadata = (sset.spectra, sset.freqs_per_spec, sset.rf_grid,
                                         sset.rf_index_map, sset.metadata)

    cut_min_idx = np.abs(fper[0] - cut_min_val).argmin()
    cut_max_idx = np.abs(fper[0] - cut_max_val).argmin()

    new_specs = []
    new_freqs = []
    new_rf_map = []
    for spec, freq, rf_vals in zip(specs, fper, rf_map):
        x = np.where(freq == 0)[0]
        for j in x:
            for i in range(2, -1, -1):
                spec[j+i] = spec[j+i+1]
                spec[j-i-1] = spec[j-i-2]


        spec = spec[cut_min_idx:cut_max_idx]
        freq = freq[cut_min_idx:cut_max_idx]
        rf_vals = rf_vals[cut_min_idx:cut_max_idx]


        new_specs.append(spec)
        new_freqs.append(freq)
        new_rf_map.append(rf_vals)

    specs = new_specs
    fper = new_freqs
    rf = rf[cut_min_idx:cut_max_idx]
    rf_map = new_rf_map

    return SpectrumSet(
        spectra=specs,
        freqs_per_spec=fper,
        rf_grid=rf,
        rf_index_map=rf_map,
        metadata=metadata
    )
