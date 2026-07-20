from __future__ import annotations

import h5py
from types import SimpleNamespace
import numpy as np
import json
import pandas as pd
import datetime, os, sys
import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm
from pysmithchart import SmithAxes

from axion_haloscope.io_working import write_hdf5, read_qshs_hdf5_dir3



def inspect(name, obj):
    print(name)
    print("  type:", "Dataset" if isinstance(obj, h5py.Dataset) else "Group")
    if isinstance(obj, h5py.Dataset):
        print("  shape:", obj.shape)
        print("  dtype:", obj.dtype)
    if obj.attrs:
        print("  attrs:")
        for key, value in obj.attrs.items():
            print(f"    {key}: {value}")


def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}  Dataset shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}  Group")


def hdf5_to_object(h5obj):
    """
    Recursively convert an HDF5 group/file into nested Python objects.
    Datasets become NumPy arrays or scalars.
    Attributes are stored under .attrs
    """
    obj = SimpleNamespace()
    
    # Store attributes
    obj.attrs = dict(h5obj.attrs)

    for key, item in h5obj.items():
        # Make key safe as a Python attribute
        safe_key = key.replace(" ", "_").replace("-", "_")

        if isinstance(item, h5py.Dataset):
            data = item[()]
            setattr(obj, safe_key, data)
        elif isinstance(item, h5py.Group):
            setattr(obj, safe_key, hdf5_to_object(item))

    return obj


def flatten_to_kv(rows_list, source_name):
    """
    Flattens multiple rows of data into one row.
    Makes data more presentable.
    """
    flat = []
    for row in rows_list:
        field = row.pop("FIELD")
        row.pop("SOURCE", None)
        value_str = ", ".join(f"{k}={v}" for k, v in row.items())
        flat.append({"SOURCE": source_name, "FIELD": field, "VALUE": value_str})
    return flat


def main():
    # ===================
    # Initialising
    # ===================

    input_dir = "input/Jan"

    pattern = "*.hdf5"
    save_fox_h5 = True

    out_root = "output/qshs_to_FOX"
    os.makedirs(out_root, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    run_dir = f"{out_root}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # =============================
    # Create complete SpectrumSet
    # =============================

    sset = read_qshs_hdf5_dir3(
        input_dir,
        pattern=pattern,
        use_shifted_frequency=True,
        sort_frequency=True,
        run_dir=run_dir,
    )

    print(f"[QSHS] Successfully loaded {sset.n_spectra()}/ {len(sset.metadata['invalid_files']) + sset.n_spectra()} spectra, {len(sset.metadata['invalid_files'])} files are invalid.")
    print(
        f"[QSHS] Shifted frequency span: "
        f"{sset.rf_grid[0]/1e6:.6f} to {sset.rf_grid[-1]/1e6:.6f} MHz"
    )

    if save_fox_h5:
        out_h5 = f"{run_dir}/converted_spectra.h5"
        write_hdf5(sset, out_h5)
        print(f"[QSHS] Saved FOX-native HDF5: {out_h5}")

    
    # ============================
    # Optional Plotting / Saving
    # ============================

    plot_qshs_data = False
    plot_slow_controls = False
    save_slow_controls = False

    valid_files = list(sset.metadata["file_name"])
    res_freqs_list = []
    cw_freqs_list = []
    bandwidths_list = []
    q_loaded_list = []

    if plot_qshs_data or save_slow_controls or plot_slow_controls:
    
        for s in tqdm(valid_files, desc="Optional: Obtaining slow controls"):
            
            with h5py.File(f"{input_dir}/{s}", "r") as f:

                power_spectra   = f["Power_Spectra"][()]
                raw_data        = f["Raw_Data"][()]
                narrow_vna_scan = f["Narrow_VNA_Scan"][()]
                wide_vna_scan   = f["Wide_VNA_Scan"][()]
                slow_controls_hardware   = f["Slow_Controls/Hardware"][()]
                slow_controls_mode_fit   = f["Slow_Controls/Mode-Fit"][()]
                slow_controls_status   = f["Slow_Controls/Status"][()]
                slow_controls_temperatures   = f["Slow_Controls/Temperatures"][()]


                # --- Hardware ---
                hardware_data = json.loads(slow_controls_hardware.decode("utf-8"))
                hardware_rows = []
                for entry in hardware_data["Hardware_Components"]:
                    for component_name, props_list in entry.items():
                        for props in props_list:
                            hardware_rows.append({"SOURCE": "Hardware", "FIELD": component_name, **props})

                # --- Mode Fit ---
                raw = slow_controls_mode_fit.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                mode_fit_data = json.loads(raw)    
                modefit_rows = []
                if plot_slow_controls:
                    fig, axes = plt.subplots(4, 1, figsize=(19, 19))

                j = 0
                for key, value in mode_fit_data.items():

                    if isinstance(value, list):

                        if plot_slow_controls:    
                            axes[j].plot(value, label=key)
                            axes[j].grid(True)
                            axes[j].legend()

                            modefit_rows.append({
                                "SOURCE": "Mode-Fit",
                                "FIELD": key,
                                "VALUE": f"[len={len(value)}] first={value[0]:.4g}, last={value[-1]:.4g}"
                            })
                            j += 1
                    else:
                        modefit_rows.append({"SOURCE": "Mode-Fit", "FIELD": key, "VALUE": value})
                        if key == "res_freq":
                            res_freqs_list.append(value)
                        elif key == "cw_freq":
                            cw_freqs_list.append(value)
                        elif key == "bandwidth":
                            bandwidths_list.append(value)
                        elif key == "q_loaded":
                            q_loaded_list.append(value)
        
                # --- Status ---
                raw = slow_controls_status.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                status_data = json.loads(raw)

                status_rows = []
                for channel, (freq, status, power) in status_data.items():
                    status_rows.append({
                        "SOURCE": "Status",
                        "FIELD": channel,
                        "Frequency (Hz)": freq,
                        "Output": status.strip(),
                        "Power (dBm)": power,
                    })

                flat_rows = (
                    flatten_to_kv(hardware_rows, "Hardware") +
                    modefit_rows +
                    flatten_to_kv(status_rows, "Status")
                )
                df = pd.DataFrame(flat_rows)

                table_str = tabulate(df, headers="keys", tablefmt="plain", showindex=False, stralign="left", numalign="left")

                # Optional: Plot slow controls
                if plot_slow_controls:
                    plt.tight_layout()
                    plt.savefig(f"{run_dir}/qshs_slow_controls_data_{valid_files.index(s)}.png", dpi=150)
                    plt.close(fig)

                # Optional: Save slow controls
                if save_slow_controls:
                    with open(f"{run_dir}/slow_controls_summary_{valid_files.index(s)}.txt", "w") as f:
                        f.write(table_str)

                        if plot_qshs_data:
                            fig, axes = plt.subplots(2, 2, figsize=(19, 9))

                # Optional: Plot QSHS data
                if plot_qshs_data:
                    fig, axes = plt.subplots(2, 2, figsize=(19, 9))
                    axes[0, 0].plot(raw_data[0],raw_data[1], label="Raw")
                    axes[0, 0].set_title("Raw Data")
                    axes[0, 0].set_xlabel("Sample index")
                    axes[0, 0].set_ylabel("ADC value")
                    axes[0, 0].grid(True)
                    axes[0, 0].legend()

                    order = np.argsort(power_spectra[0])
                    power_spectra[0] = power_spectra[0][order]
                    power_spectra[1] = power_spectra[1][order]

                    axes[1, 0].plot(power_spectra[0],power_spectra[1], label="Power Spectra")
                    axes[1, 0].set_title("Power Spectra")
                    axes[1, 0].set_xlabel("Frequency bin")
                    axes[1, 0].set_ylabel("Power")
                    axes[1, 0].grid(True)
                    axes[1, 0].legend()

                    
                    a = axes[0, 1].scatter(narrow_vna_scan[1],narrow_vna_scan[2], c=narrow_vna_scan[0], cmap='viridis', s=100)
                    axes[0, 1].set_title("Narrow VNA Scan")
                    axes[0, 1].set_xlabel("Unknown")
                    axes[0, 1].set_ylabel("Unknown")
                    axes[0, 1].set_xlim(-0.5, 0.5)
                    axes[0, 1].set_ylim(-0.5, 0.5)
                    axes[0, 1].grid(True)
                    fig.colorbar(a, ax=axes[0, 1])

                    b = axes[1, 1].scatter(wide_vna_scan[1],wide_vna_scan[2], c=wide_vna_scan[0], cmap='viridis', s=100)
                    axes[1, 1].set_title("Wide VNA Scan")
                    axes[1, 1].set_xlabel("Unknown")
                    axes[1, 1].set_ylabel("Unknown")
                    axes[0, 1].set_xlim(-0.5, 0.5)
                    axes[0, 1].set_ylim(-0.5, 0.5)
                    axes[1, 1].grid(True)
                    fig.colorbar(b, ax=axes[1, 1])
                    
                    plt.tight_layout()
                    plt.savefig(f"{run_dir}/qshs_data_{valid_files.index(s)}.png", dpi=150)
                    plt.close(fig)


                    freq = narrow_vna_scan[0]
                    s11_narrow = narrow_vna_scan[1] + 1j * narrow_vna_scan[2]
                    s11_narrow *= 50

                    s11_wide = wide_vna_scan[1] + 1j * wide_vna_scan[2]
                    s11_wide *= 50

                    fig = plt.figure(figsize=(12, 6))
                    ax1 = fig.add_subplot(1, 2, 1, projection='smith')
                    ax2 = fig.add_subplot(1, 2, 2, projection='smith')

                    sc1 = ax1.scatter(s11_narrow.real, s11_narrow.imag, marker=".", c=freq, cmap='viridis', s=10)
                    fig.colorbar(sc1, ax=ax1, label="Frequency (Hz)")
                    ax1.set_title("Narrow VNA Scan")

                    sc2 = ax2.scatter(s11_wide.real, s11_wide.imag, marker=".", c=freq, cmap='viridis', s=10)
                    fig.colorbar(sc2, ax=ax2, label="Frequency (Hz)")
                    ax2.set_title("Wide VNA Scan")

                    plt.savefig(f"{run_dir}/smith_chart_{valid_files.index(s)}.png", dpi=150)
                    plt.close(fig)



if __name__ == "__main__":
    main()