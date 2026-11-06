from __future__ import annotations
import datetime
from datetime import timedelta
import midas.file_reader
import h5py
import matplotlib.pyplot as plt
import threading
from pathlib import Path
import queue 
from typing import List, Tuple, Optional
import numpy as np
from numpy.fft import fft, rfft
FILENAME="/work/axion_data/test_runs/run00018.mid.lz4"
 

#putting these classes in for now because importing wasn't working
from dataclasses import dataclass
@dataclass
class SpectrumMetadata:
    """
    Multi-spectrum collection of metadata and other information. Not all variable can be inputted
    as of current, and is a form of future proofing.

    Attributes
    ==========
    dates            : list of (n_bins,) strings
    file_names       : list of (n_bins,)strings
    invalid_files    : list of (invalid_files,) [file_name, reason, date] triples
    b_vals           : array of (n_bins,) floats
    temps            : array of (n_bins,) floats
    q_factors        : array of (n_bins,) floats
    res_freqs        : array of (n_bins,) floats
    cw_freqs         : array of (n_bins,) floats
    bandwidths       : array of (n_bins,) floats
    """

    dates: List[str]
    file_names: List[str]
    invalid_files: List[Tuple[str, str, str]]
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

    Attributes
    ==========
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
    sample_rate: np.float64
    bandwidths: List[np.ndarray]
    dates: List[str]

    def n_spectra(self) -> int:
        return len(self.spectra)
def _infer_bin_width(freqs_1d: np.ndarray) -> float:
    """Calculates bin width manually"""
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

class MidasFileReader:

    def __init__(self, file_path, prescale=1):
        if prescale <1:
            raise ValueError("prescale must be at least 1")
        self.prescale = prescale
        self.counter = 0

        #------------------------------------------
        # Open file and get metadata from ODB dump
        #-------------------------------------------
        self.mfile = midas.file_reader.MidasFile(file_path,use_numpy=True)
        self.stage_meta = self.load_metadata_from_file()
        self.mfile_path=str(file_path)

    def run(self):
        # Instead of while true for ongoing DA load files in
        for event in self.mfile:
            if event.header.is_midas_internal_event(): # want only data events
                continue
            
            bank_names = [b.name for b in event.banks.values()]
            # skip cryo events
            if "CRYO" in bank_names:
                continue

            # prescale
            self.counter += 1
            if self.counter % self.prescale != 0:
                continue

            parsed = self.parse_event(event)

            if parsed:
                yield parsed


    def parse_event(self, event):

        groups = {}

        for name, bank in event.banks.items():
            name = name.strip()
            idx_str = "01"
            if "01" not in groups:
                groups["01"]={}
            data = bank.data

            if name == "Si01":
                typ = "i"
                groups[idx_str][typ] = [float(x) for x in data]

            if name == "Ts01":
                typ = "t"
                if hasattr(data, "__len__") and len(data) > 0:
                    ts_val = float(data[0])
                else:
                    ts_val = float(data)

                groups[idx_str]["ts"] = ts_val
                groups[idx_str]["t"] = None

        # fix missing time axis
        for idx_str, g in groups.items():
            if g.get("t") is None:
                if "i" in g:
                    g["t"] = list(range(len(g["i"])))
                elif "q" in g:
                    g["t"] = list(range(len(g["q"])))
                else:
                    g["t"] = []

        return {
            "timestamp": event.header.timestamp,
            "waveforms": groups,
        }

    def load_metadata_from_file(self):
        # want the already stored md, not new client stuff
        meta = {}
        try:
            odb = self.mfile.get_bor_odb_dump().data
            md_store=odb["Equipment"]["DAQ DSP"]["Settings"]["01"]
            run_info=odb["Runinfo"]
            # this is only for psd
            meta = {
                    "type": md_store["type"] ,
                    "x_unit": md_store["x_unit"],
                    "y_unit": md_store["y_unit"],
                    "sample_rate": md_store["sample_rate"],
                    "title": md_store["title"],
                    "start_time": run_info["Start time"],
                    "stop_time": run_info["Stop time"],
                    "start_time_binary": run_info["Start time binary/key"]["last_written"],
                    "stop_time_binary": run_info["Stop time/key"]["last_written"],
                    "n_run": run_info["Run number"]
                }
        except Exception as e:
            print("Failed to load ODB metadata from file:")
        return meta

    def read_midas_as_sset(self,tags=None):
        spectra=[]
        freqs_per_spec=[]
        dates=[]
        times=[]
        tags=tags or {} #I still don't know if cw_freqs etc is meant to be in the md
        stage= self.stage_meta
        

        for event_data in self.run():
            group = event_data["waveforms"].get("01",{})
            if "i" not in group:
                raise ValueError("Power spectrum missing")
                i_vals = np.zeros(len(group["t"]))
            else:
                i_vals = group["i"]
            if "sample_rate" in stage:
                sample_rate =1e-6*stage.get("sample_rate",{})
                freqs = np.linspace(0,sample_rate, 8000)
            else:
                sample_rate = 0
                freqs=np.zeros(8000)

            times.append(group["t"])
            for i in range (0,2):
                freqs=np.delete(freqs,0)
                freqs=np.delete(freqs,len(freqs)-1)
                i_vals=np.delete(i_vals,0)
                i_vals=np.delete(i_vals,len(i_vals)-1)
            if sample_rate ==0:
                freqs_per_spec=np.zeros(len(freqs))
            else:
                truncatevalue = sample_rate
                freqs = freqs[freqs < truncatevalue]
                freqs_per_spec.append(freqs)
            i_vals=i_vals[0:len(freqs)]
            spectra.append(i_vals)
            dates.append(str(event_data["timestamp"]))
        x_unit=stage.get("x_unit",{})
        y_unit=stage.get("y_unit",{})
        plt.plot(freqs_per_spec[5],spectra[5],color='cornflowerblue',label=dates[5])
        plt.legend()
        plt.title("6 MHz Signal Injection")
        plt.xlabel(f"Frequency ({x_unit})")
        plt.ylabel(f"PSD ({y_unit})")
        plt.savefig("outputfig",dpi=200)

        if not spectra:
            raise ValueError("No usable power spectra found in Midas file")

        rf_grid, rf_index_map = _build_rf_grid_and_map(freqs_per_spec)

        n=len(spectra)
        bandwidths=np.full(n, sample_rate, dtype=float)
        return SpectrumSet(spectra=spectra,
            freqs_per_spec=freqs_per_spec,
            rf_grid=rf_grid,
            rf_index_map=rf_index_map,
            metadata=stage,
            sample_rate=sample_rate,
            bandwidths=bandwidths,
            dates=dates)

reader = MidasFileReader(FILENAME, prescale=1)
spectrum_set = reader.read_midas_as_sset()


def write_midas_to_hdf5(sset: SpectrumSet, 
        path: str,
        compression: str | None = "gzip",
        compression_opts: int = 4
        ) -> None:
    """
    Save the Midas SpectrumSet to HDF5. Metadata written to separate group.
    """
    path = Path(path)
    with h5py.File(path, "w") as h5:
        n_spec = sset.n_spectra()
        vlen_f64 = h5py.vlen_dtype(np.dtype("float64"))
        vlen_i64 = h5py.vlen_dtype(np.dtype("int64"))

        # power and frequency data

        d_specs = h5.create_dataset("spectra", (n_spec,), dtype=vlen_f64)
        d_freqs = h5.create_dataset("freqs_per_spec", (n_spec,), dtype=vlen_f64)
        d_rfmap = h5.create_dataset("rf_index_map", (n_spec,), dtype=vlen_i64)
        h5.create_dataset("rf_grid", data=np.asarray(sset.rf_grid, np.float64),
                          compression=compression, compression_opts=compression_opts)
        # metadata
        metadata_group = h5.create_group("metadata")

        #midas specific metadata
        d_x_unit = metadata_group.create_dataset("x_unit", (n_spec), dtype=h5py.string_dtype())
        d_y_unit = metadata_group.create_dataset("y_unit", (n_spec), dtype=h5py.string_dtype())
        d_sample_rate = metadata_group.create_dataset("sample_rate", (n_spec), dtype=np.float64)
        d_type = metadata_group.create_dataset("type", (n_spec), dtype=h5py.string_dtype())

        #FOX metadata
        d_dates = metadata_group.create_dataset("dates", (n_spec,), dtype=h5py.string_dtype())
        d_file_names = metadata_group.create_dataset("file_names", (n_spec,), dtype=h5py.string_dtype())
        d_b_vals = metadata_group.create_dataset("b_vals", (n_spec,), dtype=np.float64)
        d_temps = metadata_group.create_dataset("temps", (n_spec,), dtype=np.float64)
        d_q_factors = metadata_group.create_dataset("q_factors", (n_spec,), dtype=np.float64)
        d_res_freqs = metadata_group.create_dataset("res_freqs", (n_spec,), dtype=np.float64)
        d_cw_freqs = metadata_group.create_dataset("cw_freqs", (n_spec,), dtype=np.float64)
        d_bandwidths = metadata_group.create_dataset("bandwidths", (n_spec,), dtype=np.float64)      
        start = sset.metadata.get("start_time")
        start = datetime.datetime.strptime(start, '%a %b %d %H:%M:%S %Y')
        start_binary = sset.metadata.get("start_time_binary")
        for i in range(n_spec):
            d_specs[i] = np.asarray(sset.spectra[i], np.float64)
            d_freqs[i] = np.asarray(sset.freqs_per_spec[i], np.float64)
            d_rfmap[i] = np.asarray(sset.rf_index_map[i], np.int64)
            d_x_unit = sset.metadata.get("x_unit",{})
            d_y_unit = sset.metadata.get("y_unit",{})
            d_sample_rate =sset.metadata.get("sample_rate",{})
            d_type = sset.metadata.get("type",{})
            binary_ddates=np.asarray(sset.dates[i], np.int64)
            timedif=int(binary_ddates-start_binary)
            ddates = start + datetime.timedelta(seconds=timedif)
            ddates=ddates.strftime('%Y-%m-%d %H:%M:%S')
            d_dates[i] = str(ddates)
            d_file_names[i] = str(path)
            d_b_vals[i] = 0
            d_temps[i] = 0
            d_q_factors[i] = 0
            d_res_freqs[i] = 29.96e3
            d_cw_freqs[i] = 6
            d_bandwidths[i] = sset.bandwidths[i]

        n_invalid = 0
        d_invalid_files = metadata_group.create_dataset(
            "invalid_files", (n_invalid, 3), dtype=h5py.string_dtype()
        )
    

new_hdf5=write_midas_to_hdf5(sset=spectrum_set, path="/work/robynw/packages/RAPID/midas_reader_analysis/THISONE/FOX/midas_data/run_18.hdf5")

import h5py

def dump_hdf5_structure(name, obj):
    """Callback function to print the layout of the HDF5 file."""
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name} (shape={obj.shape}, dtype={obj.dtype})")

    # Print attributes (metadata) if they exist
    if obj.attrs:
        for attr_name, attr_val in obj.attrs.items():
            print(f"{indent}    🏷️ Attr: {attr_name} = {attr_val}")

# Open the file in read-only mode
#filename = "/work/axion_data/test_runs/midas_hdf5/run_18.hdf5"
#with h5py.File(filename, "r") as f:
 #   print(f"Root structure for {filename}:")
  #  f.visititems(dump_hdf5_structure)

