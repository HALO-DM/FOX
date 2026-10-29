# For data tagging of code
# python scripts/data_analysis/data_tagging_yaml.py configs/data_tagging.yaml

from __future__ import annotations
import argparse
import matplotlib
import h5py
from types import SimpleNamespace
import numpy as np
import pandas as pd
import datetime, os, sys
import json
from pathlib import Path
from tabulate import tabulate
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict
import pathlib
import matplotlib.cm as cm
from tqdm import tqdm
import shutil
import hickle as hkl
from dataclasses import dataclass


from axion_haloscope.simulation import simulate_spectra
from axion_haloscope.io_working import SpectrumSet, read_qshs_hdf5_dir, read_hdf5, write_hdf5, read_qshs_hdf5
from axion_haloscope.convert_qshs_to_FOX import inspect, print_hdf5_structure, hdf5_to_object, flatten_to_kv, main
from axion_haloscope.load_data import load_data
from axion_haloscope.data_quality_working import power_too_high, small_bandwidth, metadata_is_zeros, too_noisy
from axion_haloscope.utils                import create_directory, find_project_root, load_yaml_config


# -----------------------------------
# Set up the objects we want to use
# -----------------------------------

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
class convspecset:
    """
    Collection of spectra and their corresponding frequencies after being converted to readable objects
    with non-zero power spectra (ideally)
    """

    spectra: List[np.ndarray]
    freqs_per_spec: List[np.ndarray]
    bandwidths: List[float]
    missing_metadata: List[int]
    missing_bandwidth: List[int]
    missing_b_vals: List[int]
    missing_res_freqs: List[int]
    missing_q_load: List[int]
    missing_temps: List[int]
    files: List[Path]
   
# ----------------------------------------------------
# Convert to readable spectra, find missing metadata
# ----------------------------------------------------

def read_dir_convert(directory: str |Path, *, pattern: str = "*.hdf5",
    power_path: str = "/Power_Spectra",
   ) -> convspecset:
    directory=Path(directory)
    files=sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No QSHS HDF5 files matching {pattern} in {directory}")

    spectra = []
    freqs_per_spec = []
    bandwidths = []
    missing_metadata = []
    missing_bandwidth = []
    missing_b_vals = []
    missing_res_freqs = []
    missing_q_load = []
    missing_temps = []

    for fp in tqdm(files, desc="Converting Spectra"):
        with h5py.File(fp, "r") as h5:
            power_spectra = h5[power_path][()]
            spectrum = power_spectra[1]
            frequencies = power_spectra[0]
            spectra.append(spectrum)
            freqs_per_spec.append(frequencies)
            try:
                raw = h5["Slow_Controls/Mode-Fit"][()].item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                mode_fit_data = json.loads(raw)
                if not isinstance(mode_fit_data, dict):
                    raise ValueError("Mode-Fit JSON must be an object")
            except (KeyError, AttributeError, TypeError, ValueError, json.JSONDecodeError):
                missing_metadata.append(1)
                missing_bandwidth.append(1)
                missing_b_vals.append(1)
                missing_res_freqs.append(1)
                missing_q_load.append(1)
                bandwidths.append(0.0)
            else:
                missing_metadata.append(0)
                missing_bandwidth.append(int("bandwidth" not in mode_fit_data))
                missing_b_vals.append(int("b_val" not in mode_fit_data))
                missing_res_freqs.append(int("res_freq" not in mode_fit_data))
                missing_q_load.append(int("q_loaded" not in mode_fit_data))
                try:
                    bandwidths.append(float(mode_fit_data.get("bandwidth", 0.0)))
                except (TypeError, ValueError):
                    bandwidths[-1:] = [0.0]
                    missing_bandwidth[-1] = 1
            missing_temps.append(int("Temperatures" not in h5["Slow_Controls"]))
    
    return convspecset(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        bandwidths=bandwidths,
        missing_metadata=missing_metadata,
        missing_bandwidth=missing_bandwidth,
        missing_b_vals=missing_b_vals,
        missing_res_freqs=missing_res_freqs,
        missing_temps=missing_temps,
        missing_q_load=missing_q_load,
        files=files,
    )

# -----------------------------------
# Main running process
# -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate haloscope run from YAML config")
    ap.add_argument("config", help="Path to YAML config (e.g. configs/simulate_run.yaml)")
    args = ap.parse_args()
    
    cfg_path = pathlib.Path(args.config).resolve()
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    
    cfg = load_yaml_config(cfg_path)
    inp, sim, inj, qc, alg, base, rb, det, out, diag = (cfg[k] for k in ("input","simulation","injection","quality",
                                                                         "alignment","baseline","rebin","detection","output","diagnostic"))

    
    project_root = find_project_root(pathlib.Path(__file__).resolve())
 
    out_root = project_root / out["root"] / "data_analysis"
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    run_dir = create_directory(out_root, f'{out["subdir_prefix"]}_{timestamp}')
    main_plots_dir = create_directory(run_dir, "main_plots")
    data_dir = create_directory(run_dir, "data")

    if diag["run_diagnostics"]:
        print("Diagnostic Mode On")
        diagnostic_mode = True
        diag_run_dir = create_directory(run_dir, "diagnostic_plots")
    else:
        diagnostic_mode = False

    # ---------------------------------------------------------
    # Data input
    # ---------------------------------------------------------

    converted=read_dir_convert(inp["directory"])
    s,f=converted.spectra,converted.freqs_per_spec
    md=SimpleNamespace(bandwidths=np.asarray(converted.bandwidths, dtype=float))
    files=converted.files

    # missing metadata from the input data

    missing_metadata=converted.missing_metadata
    missing_bandwidth=converted.missing_bandwidth
    missing_b_vals=converted.missing_b_vals # I'm not sure that any of the spectra actually have b vals
    missing_res_freqs=converted.missing_res_freqs
    missing_temps=converted.missing_temps
    missing_q_load=converted.missing_q_load

    # ----------------------------------------------
    # Set up list of errors and an error directory
    # ----------------------------------------------
    
    os.mkdir(f"output/error_files/error_files_{timestamp}")
    os.mkdir(f"output/error_files/bit_files_{timestamp}")

    # number of non metadata errors

    zero_power_tot=0
    high_power_tot=0
    total=0
    noise_tot=0
    small_bandwidth_tot=0

    # missing metadata errors

    metazeroes_tot=0
    missing_metadata_tot=0
    missing_bandwidth_tot=0
    missing_b_vals_tot=0
    missing_res_freqs_tot=[]
    missing_temps_tot=0
    missing_some_metadata_tot=0
    missing_q_tot=0
    
    # -----------------------------------------
    # Tag the data with errors
    # -----------------------------------------

    for i in tqdm(range(len(files)), desc="Tagging Spectra"):
        
        i=i
        sarray=s[i]
        split4097=[]
        freq4097=[]
        split8192=[]
        freq8192=[]
        if len(sarray)== 4097:
            split4097=sarray
            bits=[]
            freq4097=f[i]

            # non metadata related

            power_all_zeroes= not np.any(split4097)
            zero_power_bit = int(power_all_zeroes)
            high_power_bit = int(power_too_high(split4097, freq4097, md, i, p_max=qc["p_max"]))
            noise_bit = int(too_noisy(split4097, freq4097, md, i,rms_max=qc["rms_max"],nan_fail=qc["nan_fail"],robust=qc["robust"]))
            
            # missing metadata

            if missing_metadata[i]==0 and zero_power_bit==0:
                small_bandwidth_bit = int(small_bandwidth(split4097, freq4097, md, i,bw_min=qc["bw_min"]))
                missing_bandwidth_bit=missing_bandwidth[i]
                missing_b_vals_bit=missing_b_vals[i]
                missing_q_bit=missing_q_load[i]
                missing_res_freqs_bit=missing_res_freqs[i]
                missing_temps_bit=missing_temps[i]
            elif missing_metadata[i]==1 or zero_power_bit==1:
    
                small_bandwidth_bit=0
                missing_bandwidth_bit=missing_bandwidth[i]
                missing_b_vals_bit=1
                missing_q_bit=missing_q_load[i]
                missing_res_freqs_bit=missing_res_freqs[i]
                missing_temps_bit=missing_temps[i]
            missing_all_metadata=missing_metadata[i]
            missing_some_metadata=missing_bandwidth_bit+missing_res_freqs_bit+missing_q_bit+missing_temps_bit
            bits=[zero_power_bit,high_power_bit,noise_bit,small_bandwidth_bit,missing_all_metadata,missing_some_metadata]
            missing_res_freqs_tot.append(missing_res_freqs_bit)
            if np.sum(bits)>0:
                shutil.copy(f"{files[i]}", f"output/error_files/error_files_{timestamp}/{i}_{zero_power_bit}{high_power_bit}{noise_bit}{small_bandwidth_bit}{missing_all_metadata}{missing_some_metadata}")
                total+=1
            with open(f"output/error_files/bit_files_{timestamp}/{i}","w") as fi:
                    bit_str=[str(item) for item in bits]
                    writedata='\n'.join(bit_str)
                    fi.writelines(writedata)
        elif len(sarray)== 8192:
            split8192=sarray
            freq8192=f[i]
            bits=[]

            # non metadata related

            power_all_zeroes= not np.any(split8192)
            zero_power_bit = int(power_all_zeroes)
            high_power_bit = int(power_too_high(split8192, freq8192, md, i,p_max=qc["p_max"]))
            noise_bit = int(too_noisy(split8192, freq8192, md, i,rms_max=qc["rms_max"],nan_fail=qc["nan_fail"],robust=qc["robust"]))
            
            # missing metadata
            
            if missing_metadata[i]==0 and zero_power_bit==0:
                small_bandwidth_bit = int(small_bandwidth(split8192, freq8192, md, i,bw_min=qc["bw_min"]))
                
                missing_bandwidth_bit=missing_bandwidth[i]
                missing_b_vals_bit=missing_b_vals[i]
                missing_q_bit=missing_q_load[i]
                missing_res_freqs_bit=missing_res_freqs[i]
                missing_temps_bit=missing_temps[i]
            elif missing_metadata[i]==1 or zero_power_bit==1:

                small_bandwidth_bit=0
                missing_bandwidth_bit=missing_bandwidth[i]
                missing_b_vals_bit=1
                missing_q_bit=missing_q_load[i]
                missing_res_freqs_bit=missing_res_freqs[i]
                missing_temps_bit=missing_temps[i]
            
            missing_all_metadata=missing_metadata[i]
            missing_some_metadata=missing_bandwidth_bit+missing_res_freqs_bit+missing_q_bit+missing_temps_bit
            bits=[zero_power_bit,high_power_bit,noise_bit,small_bandwidth_bit,missing_all_metadata,missing_some_metadata]
            missing_res_freqs_tot.append(missing_res_freqs_bit)

            if np.sum(bits)>0:
                shutil.copy(f"{files[i]}", f"output/error_files/error_files_{timestamp}/{i}_{zero_power_bit}{high_power_bit}{noise_bit}{small_bandwidth_bit}{missing_all_metadata}{missing_some_metadata}")
                
                total+=1
            with open(f"output/error_files/bit_files_{timestamp}/{i}","w") as fi:
                    bit_str=[str(item) for item in bits]
                    writedata='\n'.join(bit_str)
                    fi.writelines(writedata)
        else:
            print(f"File {files[i]} is empty or has the wrong shape, skipping tagging.")
        zero_power_tot=zero_power_tot+zero_power_bit
        high_power_tot=high_power_tot+high_power_bit
        noise_tot=noise_tot+noise_bit
        small_bandwidth_tot=small_bandwidth_tot+small_bandwidth_bit
        missing_metadata_tot=missing_metadata_tot+missing_all_metadata
        missing_some_metadata_tot=missing_some_metadata_tot+missing_some_metadata
        missing_bandwidth_tot=missing_bandwidth_tot+missing_bandwidth_bit
        missing_b_vals_tot=missing_b_vals_tot+missing_b_vals_bit
        missing_q_tot=missing_q_tot+missing_q_bit
        missing_temps_tot=missing_temps_tot+missing_temps_bit
    
    print("-----------------------------------------")
    print(total," invalid spectra found >:(")
    print("-----------------------------------------")
    bad_runs=[["Zero power",zero_power_tot], ["High power",high_power_tot],["Too much noise",noise_tot], ["Small bandwidth",small_bandwidth_tot]]
    df=pd.DataFrame(bad_runs,columns=["Issue", "No. of spectra"])
    print(df)
    print("-----------------------------------------")
    print("Missing metadata issues:")
    print("-----------------------------------------")
    bad_runs_md=[["Missing all metadata",missing_metadata_tot],["Missing bandwidth",missing_bandwidth_tot],["Missing resonant freq",np.sum(missing_res_freqs_tot)],["Missing temps",missing_temps_tot],["Missing Loaded Q",missing_q_tot]]
    df_md=pd.DataFrame(bad_runs_md, columns=["Issue", "No of spectra"])
    print(df_md)
    print("-----------------------------------------")
    print("For file by file breakdown, see error files. Bits correspond to zero power, high power, small bandwidth, all metadata missing, and number of metadata missing")

if __name__ == "__main__":
    main()
