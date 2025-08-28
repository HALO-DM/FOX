from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import h5py

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
        bin_width = _infer_bin_width(freqs_per_spec[0])

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

def read_hdf5(path: str | Path) -> SpectrumSet:
    """
    Load SpectrumSet from HDF5 produced by write_hdf5().
    """
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

    return SpectrumSet(
        spectra=spectra_list,
        freqs_per_spec=freqs_list,
        rf_grid=rf_grid,
        rf_index_map=rf_index_map,
    )
