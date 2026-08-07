from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import h5py, json, sys
import os
from tqdm import tqdm

@dataclass
class SpectrumMetadata:
    """
    Metadata innit

    dates            : list of floats
    file_names       : list of strings
    invalid_files    : list of [file_name, reason, date] triples
    b_vals           : list of floats
    temps            : list of floats
    q_factors        : list of floats
    res_freqs        : list of floats
    cw_freqs         : list of floats
    bandwidths       : list of floats
    """

    dates: np.ndarray
    file_names: np.ndarray
    invalid_files: np.ndarray
    b_vals: np.ndarray
    temps: np.ndarray
    q_factors: np.ndarray
    res_freqs: np.ndarray
    cw_freqs: np.ndarray
    bandwidths: np.ndarray


    
@dataclass
class SpectrumSet:
    """
    Multi-spectrum scan on a common RF grid.

    spectra        : list of (n_bins_i,) float arrays (raw power)
    freqs_per_spec : list of (n_bins_i,) float arrays [Hz]
    rf_grid        : (N_rf,) float array [Hz]
    rf_index_map   : list of (n_bins_i,) int arrays mapping each spectrum into rf_grid
    metadata       : SpectrumMetadata object containing metadata 
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
                       rf_index_map=rf_index_map,
                       metadata=None)

def write_npz(sset: SpectrumSet, path: str | Path) -> None:
    path = Path(path)
    spectra = np.stack(sset.spectra, axis=0)
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
                       rf_index_map=rf_index_map,
                       metadata=None)


# ----------------------------
# HDF5 I/O
# ----------------------------

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
        metadata_group = h5.create_group("metadata")
        
        d_dates = metadata_group.create_dataset("dates", (n_spec,), dtype=h5py.string_dtype())
        d_file_names = metadata_group.create_dataset("file_names", (n_spec,), dtype=h5py.string_dtype())
        d_b_vals = metadata_group.create_dataset("b_vals", (n_spec,), dtype=np.float64)
        d_temps = metadata_group.create_dataset("temps", (n_spec,), dtype=np.float64)
        d_q_factors = metadata_group.create_dataset("q_factors", (n_spec,), dtype=np.float64)
        d_res_freqs = metadata_group.create_dataset("res_freqs", (n_spec,), dtype=np.float64)
        d_cw_freqs = metadata_group.create_dataset("cw_freqs", (n_spec,), dtype=np.float64)
        d_bandwidths = metadata_group.create_dataset("bandwidths", (n_spec,), dtype=np.float64)

        for i in range(n_spec):
            d_specs[i] = np.asarray(sset.spectra[i], np.float64)
            d_freqs[i] = np.asarray(sset.freqs_per_spec[i], np.float64)
            d_rfmap[i] = np.asarray(sset.rf_index_map[i], np.int64)
            d_dates[i] = sset.metadata.dates[i]
            d_file_names[i] = sset.metadata.file_names[i]
            d_b_vals[i] = sset.metadata.b_vals[i]
            d_temps[i] = sset.metadata.temps[i]
            d_q_factors[i] = sset.metadata.q_factors[i]
            d_res_freqs[i] = sset.metadata.res_freqs[i]
            d_cw_freqs[i] = sset.metadata.cw_freqs[i]
            d_bandwidths[i] = sset.metadata.bandwidths[i]

        # invalid_files handled separately, sized to its own length
        n_invalid = len(sset.metadata.invalid_files)
        d_invalid_files = metadata_group.create_dataset(
            "invalid_files", (n_invalid, 3), dtype=h5py.string_dtype()
        )
        for k, row in enumerate(sset.metadata.invalid_files):
            # row = [file_name, reason, date]
            d_invalid_files[k, :] = [str(row[0]), str(row[1]), str(row[2])]


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

            meta_group = h5["metadata"]

            # dates / file_names: scalar strings per spectrum
            dates = [s.decode() if isinstance(s, bytes) else s
                     for s in meta_group["dates"][:]]
            file_names = [s.decode() if isinstance(s, bytes) else s
                          for s in meta_group["file_names"][:]]

            # b_vals / temps / q_factors / res_freqs / cw_freqs / bandwidths:
            # scalar float per spectrum -> flat float64 arrays, direct read
            b_vals = np.asarray(meta_group["b_vals"][:], np.float64)
            temps = np.asarray(meta_group["temps"][:], np.float64)
            q_factors = np.asarray(meta_group["q_factors"][:], np.float64)
            res_freqs = np.asarray(meta_group["res_freqs"][:], np.float64)
            cw_freqs = np.asarray(meta_group["cw_freqs"][:], np.float64)
            bandwidth = np.asarray(meta_group["bandwidths"][:], np.float64)

            # invalid_files: list of strings
            raw_invalid = meta_group["invalid_files"][:]  # shape (n_invalid, 3)
            invalid_files = [
                [c.decode() if isinstance(c, bytes) else c for c in row]
                for row in raw_invalid
            ]

            metadata = SpectrumMetadata(
                        dates=dates,
                        file_names=file_names,
                        invalid_files=invalid_files,
                        b_vals=b_vals,
                        temps=temps,
                        q_factors=q_factors,
                        res_freqs=res_freqs,
                        cw_freqs=cw_freqs,
                        bandwidths=bandwidth
                    )

            return SpectrumSet(
                spectra=spectra_list,
                freqs_per_spec=freqs_list,
                rf_grid=rf_grid,
                rf_index_map=rf_index_map,
                metadata=metadata,
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"HDF5 file not found: {path}. Please run the QSHS conversion "
                                "script first to generate the necessary HDF5 file.")


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
        slow_controls_status  = h5["Slow_Controls/Status"][()]

        file_name_str = h5.attrs["This_File"]
        if isinstance(file_name_str, bytes):
            file_name_str = file_name_str.decode("utf-8")

        date_str = h5.attrs["Date-Time"]
        if isinstance(date_str, bytes):
            date_str = date_str.decode("utf-8")

        raw = slow_controls_mode_fit.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        mode_fit_data = json.loads(raw)

        raw = slow_controls_status.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        status_data = json.loads(raw)
        
        for key, value in mode_fit_data.items():
            if key == "res_freq":
                res_freq_str = (value)
            elif key == "bandwidth":
                bandwidths_str = value
            elif key == "q_loaded":
                q_loaded_str = value

            # Add more items here later when more get added from QSHS data set

        for key, value in status_data.items():
            if key == "RF1":
                cw_freq_str = value[0]

        spec_metadata = SpectrumMetadata(
            dates=date_str,
            file_names=file_name_str,
            invalid_files= None,
            b_vals=None,
            temps=None,
            q_factors=q_loaded_str,
            res_freqs=res_freq_str,
            cw_freqs=cw_freq_str,
            bandwidths=bandwidths_str,

        )

    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(
            f"Expected {power_path} to have shape (2, n_bins), got {arr.shape}"
        )

    freq_offset_hz = np.asarray(arr[0], dtype=float)
    power = np.asarray(arr[1], dtype=float)

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

    Returns one merged SpectrumSet containing all spectra, with metadata as a dictionary.
    """

    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No QSHS HDF5 files matching {pattern} in {directory}")
    
    spectra = []
    freqs_per_spec = []
    invalid_files = []
    dates = []
    file_names = []
    invalid_file = []
    b_vals = []
    temps = []
    q_factors = []
    res_freqs = []
    cw_freqs = []
    bandwidth = []

    for fp in tqdm(files, desc="Processing spectra"):
        
        try:    
            one = read_qshs_hdf5(
                fp,
                power_path=power_path,
                use_shifted_frequency=use_shifted_frequency,
                center_frequency_hz=center_frequency_hz,
                sort_frequency=sort_frequency,
            )

            if np.all(one.spectra[0] == 0):
                invalid_files.append([one.metadata.file_names, "power spectra is zeros", one.metadata.dates])
            else:
                spectra.append(one.spectra[0])
                freqs_per_spec.append(one.freqs_per_spec[0])
                spectrum_metadata = one.metadata
                dates.append(spectrum_metadata.dates)
                file_names.append(spectrum_metadata.file_names)
                invalid_file.append(spectrum_metadata.invalid_files)
                b_vals.append(spectrum_metadata.b_vals)
                temps.append(spectrum_metadata.temps)
                q_factors.append(spectrum_metadata.q_factors)
                res_freqs.append(spectrum_metadata.res_freqs)
                cw_freqs.append(spectrum_metadata.cw_freqs)
                bandwidth.append(spectrum_metadata.bandwidths)

                
        except json.decoder.JSONDecodeError:
            invalid_files.append([one.metadata.file_names, "metadata is missing", one.metadata.dates])
            continue

    metadata = SpectrumMetadata(
            dates=dates,
            file_names=file_names,
            invalid_files=invalid_files,
            b_vals=b_vals,
            temps=temps,
            q_factors=q_factors,
            res_freqs=res_freqs,
            cw_freqs=cw_freqs,
            bandwidths=bandwidth
        )
    
    rf_grid, rf_index_map = _build_rf_grid_and_map(
        freqs_per_spec,
        bin_width=bin_width,
    )

    return SpectrumSet(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        rf_grid=rf_grid,
        rf_index_map=rf_index_map,
        metadata=metadata
    )