"""
Load Data
=========

Module to manage loading data for `data_analysis.py`
"""

from pathlib import Path
from typing import Dict


from axion_haloscope.io_working import SpectrumSet, read_qshs_hdf5_dir, read_hdf5, write_hdf5
from axion_haloscope.simulation import simulate_spectra

def load_data(
    input_mode: str,
    diagnostic_mode: bool,
    read_dir: Path,
    file_name: str,
    save_dir: Path,
    sim: Dict,
    inj: Dict,
    out: Dict
) -> SpectrumSet:
    """
    Loads data in one of 3 ways: directly from source files, an already converted set of files,
    or generates a basic simulation. More physical simulation hasn't been integrated yet.

    Parameters
    ==========
    input_mode: string
        Which input to use
    diagnostic_mode: bool
        If diagnostic mode is enabled, prints extra console text
    read_dir: Path
        Directory from which the data is to be read from
    file_name: string
        Name of data file to be saved
    save_dir: Path
        Directory where converted data is saved.
    sim: Dict
        Dictionary containing all simulation settings from YAML
    inj: Dict
        Dictionary containing all injection settings from YAML
    out: Dict
        Dictionary containing all output settings from YAML

    Returns
    ======
    sset: SpectrumSet
        SpectrumSet data structure containing stacked spectra, metadata and extra information used
        in the code later

    Raises
    ======
    ValueError
        If input isn't one of the following: 'read_data', 'convert_data', or 'simulation'.
        These are the only available options for input currently.
    """
    if input_mode == "read_data":
        if diagnostic_mode:
            print("=" * 60)
            print("Data Reading")
            print("=" * 60)
        # 1) Read in Data
        input_file_name = file_name
        sset = read_hdf5(f"{read_dir}/{input_file_name}")

    elif input_mode == "convert_data":
        if diagnostic_mode:
            print("=" * 60)
            print("Converting Data")
            print("=" * 60)

        # 1) Convert QSHS data to FOX
        sset = read_qshs_hdf5_dir(
                read_dir,
                pattern="*.hdf5",
                use_shifted_frequency=True,
                sort_frequency=True,
            )

    elif input_mode == "simulation":
        if diagnostic_mode:
            print("=" * 60)
            print("Running Simulation")
            print("=" * 60)
        # 1) Simulate
        sset = simulate_spectra(
        n_spectra=sim["n_spectra"], n_bins=sim["n_bins"],
        bin_width_hz=sim["bin_width_hz"], f_start_hz=sim["f_start_hz"],
        tune_step_bins=sim["tune_step_bins"], rng_seed=sim["rng_seed"],
        noise_sigma=sim["noise_sigma"], injected_axion=inj
        )
    else:
        raise ValueError(f"Input Mode '{input_mode}' not recognised. Pleaese make sure you have "
                         "selected 'read_data', 'convert_data', or 'simulation'. Note: if you "
                         "have selected 'simualtion', please make sure that the simulation "
                         "information is filled out.")
    if out["save_data"]:
        out_h5 = f"{save_dir}/converted_spectra.h5"
        write_hdf5(sset, out_h5)
        print(f"[QSHS] Saved FOX-native HDF5: {out_h5}")

    return sset
