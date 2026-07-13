from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import h5py, json, sys
import os

@dataclass
class SpectrumMetadata:
    """
    Metadata innit

    date            : list of floats
    file_name       : list of strings
    b_vals          : list of floats
    temps           : list of floats
    q_factor        : list of floats
    res_freq        : list of floats
    bandwidth       : list of floats
    tuning_angle    : list of floats
    volume          : list of floats
    """

    date: np.ndarray
    file_name: np.ndarray
    b_vals: np.ndarray
    temps: np.ndarray
    q_factor: np.ndarray
    res_freq: np.ndarray
    bandwidth: np.ndarray
    tuning_angle: np.ndarray
    volume: np.ndarray

    
@dataclass
class SpectrumSet:
    """
    Multi-spectrum scan on a common RF grid.

    spectra        : list of (n_bins_i,) float arrays (raw power)
    freqs_per_spec : list of (n_bins_i,) float arrays [Hz]
    rf_grid        : (N_rf,) float array [Hz]
    rf_index_map   : list of (n_bins_i,) int arrays mapping each spectrum into rf_grid
    """
    spectra: List[np.ndarray]
    freqs_per_spec: List[np.ndarray]
    rf_grid: np.ndarray
    rf_index_map: List[np.ndarray]
    metadata: SpectrumMetadata

    def n_spectra(self) -> int:
        return len(self.spectra)

def _infer_bin_width(freqs_1d: np.ndarray) -> float:
    df = np.diff(freqs_1d.astype(float, copy=False))
    df = df[np.isfinite(df)]
    return float(np.median(df)) if df.size else 0.0

def _build_rf_grid_and_map(freqs_per_spec: List[np.ndarray],
                           bin_width: Optional[float] = None,
                           tol: float = 0.25) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build a common RF grid and per-spectrum index maps.
    tol is the allowed fractional snapping error to the nearest bin.
    """
    if bin_width is None:
        bin_width = _infer_bin_width(freqs_per_spec[-1])

    # global min freq
    f0 = float(min(float(f.min()) for f in freqs_per_spec))
    idx_lists: List[np.ndarray] = []
    max_idx = 0

    for i, fi in enumerate(freqs_per_spec):
        rel = (fi - f0) / bin_width
        idx = np.rint(rel).astype(int)
        err = np.abs(rel - idx)
        if np.any(err > tol):
            bad = np.where(err > tol)[0][:3]
            raise ValueError(
                f"Frequency grid not compatible with bin width (spec {i}). "
                f"Example offending bins: {bad} (|Δ/bin|>{tol}). "
                f"Supply bin_width explicitly or relax tol."
            )
        idx_lists.append(idx)
        if idx.size:
            max_idx = max(max_idx, int(idx.max()))
    rf_grid = f0 + np.arange(max_idx + 1, dtype=float) * bin_width
    return rf_grid, idx_lists




# ----------------------------
# NPZ bundle I/O
# ----------------------------
def read_npz(npz_path: str | Path) -> SpectrumSet:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        spectra = z["spectra"]
        freqs   = z["freqs"]
        rf_grid = np.asarray(z["rf_grid"], float) if "rf_grid" in z.files else None

    spectra_list = [np.asarray(s, float) for s in spectra]
    freqs_list   = [np.asarray(f, float) for f in freqs]

    if rf_grid is None:
        rf_grid, rf_index_map = _build_rf_grid_and_map(freqs_list)
    else:
        bw = _infer_bin_width(rf_grid)
        rf_index_map = []
        for f in freqs_list:
            rel = (f - rf_grid[0]) / bw
            rf_index_map.append(np.rint(rel).astype(int))

    return SpectrumSet(spectra=spectra_list,
                       freqs_per_spec=freqs_list,
                       rf_grid=np.asarray(rf_grid, float),
                       rf_index_map=rf_index_map)

def write_npz(sset: SpectrumSet, path: str | Path) -> None:
    path = Path(path)
    spectra = np.stack(sset.spectra, axis=0)
    # freqs_per_spec may be ragged; save as object with allow_pickle=False -> not allowed.
    # So pad to max length with NaNs for a portable 2D float array.
    max_len = max(len(f) for f in sset.freqs_per_spec)
    freqs_2d = np.full((sset.n_spectra(), max_len), np.nan, float)
    for i, f in enumerate(sset.freqs_per_spec):
        freqs_2d[i, :len(f)] = f
    np.savez(path, spectra=spectra, freqs=freqs_2d, rf_grid=sset.rf_grid)



# ----------------------------
# CSV directory I/O
# ----------------------------
def read_csv_dir(csv_dir: str | Path,
                 pattern: str = "spectrum_*.csv",
                 freq_col: str = "freq_Hz",
                 power_col: str = "power",
                 bin_width: Optional[float] = None) -> SpectrumSet:
    """
    Read a directory of per-spectrum CSV files with columns [freq_Hz, power].
    Builds a common RF grid and index map automatically.
    """
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matching {pattern} in {csv_dir}")

    freqs_list, specs_list = [], []
    for fp in files:
        data = np.genfromtxt(fp, delimiter=",", names=True, dtype=None, encoding=None)
        cols = {name.lower(): name for name in data.dtype.names}
        f = np.asarray(data[cols.get(freq_col.lower(), list(cols.values())[0])], float)
        p = np.asarray(data[cols.get(power_col.lower(), list(cols.values())[1])], float)
        if f.ndim != 1 or p.ndim != 1 or f.size != p.size:
            raise ValueError(f"Malformed CSV: {fp}")
        freqs_list.append(f)
        specs_list.append(p)

    # RF grid + map
    bw = bin_width if bin_width is not None else _infer_bin_width(freqs_list[0])
    rf_grid, rf_index_map = _build_rf_grid_and_map(freqs_list, bin_width=bw)
    return SpectrumSet(spectra=specs_list,
                       freqs_per_spec=freqs_list,
                       rf_grid=rf_grid,
                       rf_index_map=rf_index_map)


# ----------------------------
# HDF5 I/O (compact + ragged-safe via vlen)
# ----------------------------
def _write_metadata_group(h5file: h5py.File, metadata: list, vlen_f64) -> None:
    """
    Write a list of per-spectrum SpectrumMetadata objects as one group,
    transposed into one dataset per field (length n_spectra each).
    """
    if not metadata:
        return
    
    grp = h5file.create_group("metadata")
    n = len(metadata)
    field_names = [f.name for f in dataclasses.fields(metadata[0])]
    for field_name in field_names:
        values = [getattr(m, field_name) for m in metadata]
        # ragged/array fields (b_vals, temps) -> vlen float64, NaN row if missing
        if any(isinstance(v, np.ndarray) for v in values):
            dset = grp.create_dataset(field_name, (n,), dtype=vlen_f64)
            for i, v in enumerate(values):
                dset[i] = (
                    np.asarray(v, dtype=np.float64)
                    if v is not None
                    else np.array([np.nan])
                )
            continue

        # string fields (date, or q_factor/res_freq/bandwidth if ever str)
        if any(isinstance(v, str) for v in values):
            str_vals = [v if v is not None else "" for v in values]
            grp.create_dataset(field_name, data=str_vals, dtype=h5py.string_dtype())
            continue

        # scalar numeric fields -> float64, NaN fill for None
        arr = np.array(
            [v if v is not None else np.nan for v in values],
            dtype=np.float64,
        )
        grp.create_dataset(field_name, data=arr)

def write_hdf5(sset: SpectrumSet, path: str | Path,
               compression: str | None = "gzip", compression_opts: int = 4) -> None:
    """
    Save SpectrumSet to HDF5 using vlen datasets for ragged arrays.
    """
    path = Path(path)
    with h5py.File(path, "w") as h5:
        n_spec = sset.n_spectra()
        vlen_f64 = h5py.vlen_dtype(np.dtype("float64"))
        vlen_i64 = h5py.vlen_dtype(np.dtype("int64"))

        d_specs = h5.create_dataset("spectra", (n_spec,), dtype=vlen_f64)
        d_freqs = h5.create_dataset("freqs_per_spec", (n_spec,), dtype=vlen_f64)
        d_rfmap = h5.create_dataset("rf_index_map", (n_spec,), dtype=vlen_i64)
        h5.create_dataset("rf_grid", data=np.asarray(sset.rf_grid, np.float64),
                          compression=compression, compression_opts=compression_opts)

        for i in range(n_spec):
            d_specs[i] = np.asarray(sset.spectra[i], np.float64)
            d_freqs[i] = np.asarray(sset.freqs_per_spec[i], np.float64)
            d_rfmap[i] = np.asarray(sset.rf_index_map[i], np.int64)

        _write_metadata_group(h5, sset.metadata, vlen_f64)

def _read_metadata_group(h5file: h5py.File) -> Dict[str, list]:
    """
    Read the 'metadata' group and return it as a dict of per-field lists,
    each of length n_spec (mirrors the dict-of-lists shape used on write).
    """
    if "metadata" not in h5file:
        return {}

    grp = h5file["metadata"]
    field_names = list(SpectrumMetadata.__dataclass_fields__.keys())

    field_values: Dict[str, list] = {}
    for field_name in field_names:
        dset = grp[field_name]
        raw = dset[()]

        if h5py.check_string_dtype(dset.dtype):
            decoded = [v.decode() if isinstance(v, bytes) else v for v in raw]
            field_values[field_name] = [v if v != "" else None for v in decoded]

        elif h5py.check_vlen_dtype(dset.dtype):
            field_values[field_name] = [None if (v.size == 1 and np.isnan(v[0])) else np.asarray(v, np.float64) for v in raw]

        else:
            field_values[field_name] = [None if np.isnan(v) else float(v) for v in raw]

    return field_values

def read_hdf5(path: str | Path) -> SpectrumSet:
    """
    Load SpectrumSet from HDF5 produced by write_hdf5().
    """
    try:
        path = Path(path)
        with h5py.File(path, "r") as h5:
            rf_grid = np.asarray(h5["rf_grid"], np.float64)
            d_specs = h5["spectra"]
            d_freqs = h5["freqs_per_spec"]
            d_rfmap = h5["rf_index_map"]

            n_spec = d_specs.shape[0]
            spectra_list: List[np.ndarray] = []
            freqs_list: List[np.ndarray] = []
            rf_index_map: List[np.ndarray] = []

            for i in range(n_spec):
                spectra_list.append(np.asarray(d_specs[i], np.float64))
                freqs_list.append(np.asarray(d_freqs[i], np.float64))
                rf_index_map.append(np.asarray(d_rfmap[i], np.int64))

            metadata = _read_metadata_group(h5)
    except FileNotFoundError:
        raise FileNotFoundError(f"HDF5 file not found: {path}. Please run the QSHS conversion "
                                "script first to generate the necessary HDF5 file.")
        
    return SpectrumSet(
        spectra=spectra_list,
        freqs_per_spec=freqs_list,
        rf_grid=rf_grid,
        rf_index_map=rf_index_map,
        metadata=metadata
    )




def read_qshs_hdf5(
    path: str | Path,
    *,
    power_path: str = "/Power_Spectra",
    use_shifted_frequency: bool = True,
    center_frequency_hz: float | None = None,
    sort_frequency: bool = True,
) -> SpectrumSet:
    """
    Read one QSHS HDF5 file and convert it to a FOX SpectrumSet.

    Current QSHS assumption:
      /Power_Spectra has shape (2, n_bins)
      row 0 = FFT frequency-offset axis [Hz]
      row 1 = power spectrum

    By default, this keeps the shifted frequency axis:
      rf_grid = frequency_offset_hz

    If center_frequency_hz is provided and use_shifted_frequency=False:
      rf_grid = center_frequency_hz + frequency_offset_hz

    Returns a SpectrumSet with one spectrum.
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        arr = np.asarray(h5[power_path][()], dtype=float)
        slow_controls_mode_fit   = h5["Slow_Controls/Mode-Fit"][()]

        file_name_str = h5.attrs["This_File"]
        if isinstance(file_name_str, bytes):
            file_name_str = file_name_str.decode("utf-8")

        # attribute lookup, not a dataset lookup
        date_str = h5.attrs["Date-Time"]
        if isinstance(date_str, bytes):
            date_str = date_str.decode("utf-8")

        raw = slow_controls_mode_fit.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        mode_fit_data = json.loads(raw)
        
        for key, value in mode_fit_data.items():
            if key == "res_freq":
                res_freq_str = (value)
            elif key == "bandwidth":
                bandwidths_str = value
            elif key == "q_loaded":
                q_loaded_str = value

        spec_metadata = SpectrumMetadata(
            date=date_str,
            file_name=file_name_str,
            b_vals=None,
            temps=None,
            q_factor=q_loaded_str,
            res_freq=res_freq_str,
            bandwidth=bandwidths_str,
            tuning_angle=None,
            volume=None,
        )

    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(
            f"Expected {power_path} to have shape (2, n_bins), got {arr.shape}"
        )

    freq_offset_hz = np.asarray(arr[0], dtype=float)
    power = np.asarray(arr[1], dtype=float)

    # QSHS row 0 appears to be FFT-ordered:
    # 0, +df, ..., +fmax, -fmax, ..., -df.
    # Sort into physical frequency order for plotting/pipeline use.
    if sort_frequency:
        order = np.argsort(freq_offset_hz)
        freq_offset_hz = freq_offset_hz[order]
        power = power[order]

    if use_shifted_frequency:
        freq_hz = freq_offset_hz
    else:
        if center_frequency_hz is None:
            raise ValueError(
                "center_frequency_hz is required when use_shifted_frequency=False"
            )
        freq_hz = float(center_frequency_hz) + freq_offset_hz

    return SpectrumSet(
        spectra=[power],
        freqs_per_spec=[freq_hz],
        rf_grid=freq_hz.copy(),
        rf_index_map=[np.arange(freq_hz.size, dtype=int)],
        metadata=spec_metadata
        )



def read_qshs_hdf5_dir(
    directory: str | Path,
    valid_files_directory: str | Path,
    *,
    pattern: str = "*.hdf5",
    power_path: str = "/Power_Spectra",
    use_shifted_frequency: bool = True,
    center_frequency_hz: float | None = None,
    sort_frequency: bool = True,
    bin_width: float | None = None,
    run_dir: str | Path | None = None,
) -> SpectrumSet:
    """
    Read a directory of QSHS HDF5 files.

    Assumes:
      - one spectrum per file
      - each file has /Power_Spectra with row 0 = frequency offset
        and row 1 = power

    Returns one merged SpectrumSet containing all spectra.
    """

    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No QSHS HDF5 files matching {pattern} in {directory}")
    

    
    with h5py.File(valid_files_directory / "valid_files.h5", "r") as h5:
        valid_file_names = [f.decode('utf-8') for f in h5["valid_files"][:]]

    spectra = []
    freqs_per_spec = []
    spec_metadata = []
    missing_modefit = []
    for fp in files:
        
        try:    
            one = read_qshs_hdf5(
                fp,
                power_path=power_path,
                use_shifted_frequency=use_shifted_frequency,
                center_frequency_hz=center_frequency_hz,
                sort_frequency=sort_frequency,
            )
            if one.metadata.file_name in valid_file_names:
                spectra.append(one.spectra[0])
                freqs_per_spec.append(one.freqs_per_spec[0])
                spec_metadata.append(one.metadata)
        except json.decoder.JSONDecodeError:
            missing_modefit.append(os.path.basename(fp))
            continue

    print(f"{len(missing_modefit)} / {len(files)}, files are unloadable as metadata is missing.")
    with open(run_dir/'unloadable_files.txt', 'w+') as f:
        f.write('Unloadable files as missing metadata')
        for file in missing_modefit:
            f.write(f"\n {str(file)}")

    # Build one common grid for all files.
    #
    # In shifted-frequency mode, all spectra likely share the same frequency-offset
    # grid. That means they will all overlap perfectly.
    #
    # Later, when using absolute RF center frequencies per file, this same helper
    # will let spectra land at different RF positions.
    rf_grid, rf_index_map = _build_rf_grid_and_map(
        freqs_per_spec,
        bin_width=bin_width,
    )

    return SpectrumSet(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        rf_grid=rf_grid,
        rf_index_map=rf_index_map,
        metadata=spec_metadata
    )
