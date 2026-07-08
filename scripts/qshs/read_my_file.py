import h5py
from types import SimpleNamespace
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import sys, datetime, os
from tabulate import tabulate
from pysmithchart import SmithAxes
from tqdm import tqdm

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


out_root = "output/qshs_spectra"
os.makedirs(out_root, exist_ok=True)


timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
run_dir = f"{out_root}/run_{timestamp}"
os.makedirs(run_dir, exist_ok=True)
spectra_list = []
res_freqs_list = []
bandwidths_list = []
q_loaded_list = []
#run_dir.mkdir(parents=True, exist_ok=True)

# Directory of larger data set
directory_all = 'input/Feb/All'
for filename in os.listdir(directory_all):
    if filename.endswith(".hdf5"):
        spectra_list.append(f"All/{filename}")


# Test data set
""" directory = 'input/Jan_QSHS'
for filename in os.listdir(directory):
    if filename.endswith(".hdf5"):
        spectra_list.append(f"{filename}")
 """
# Whether to plot the QSHS data and slow controls
plot_qshs_data = False
plot_slow_controls = False

valid_files = []
invalid_files = [] # Some aspect of the file is missing

for s in tqdm(spectra_list, desc="Processing spectra"):
    
    with h5py.File(f"input/Feb/{s}", "r") as f:
    # with h5py.File(f"input/Feb/{s}", "r") as f:
        '''f.visititems(print_hdf5_structure)
        f.visititems(inspect)
        data = hdf5_to_object(f)
        print(data.attrs)'''

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
        

        try:
            mode_fit_data = json.loads(raw)
            valid_files.append(s)
        except json.JSONDecodeError:
            invalid_files.append(s)
            continue
            

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
                elif key == "bandwidth":
                    bandwidths_list.append(value)
                elif key == "q_loaded":
                    q_loaded_list.append(value)

        if plot_slow_controls:
            plt.tight_layout()
            plt.savefig(f"{run_dir}/qshs_slow_controls_data_{spectra_list.index(s)}.png", dpi=150)
            plt.close(fig)


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

        with open(f"{run_dir}/slow_controls_summary_{spectra_list.index(s)}.txt", "w") as f:
            f.write(table_str)

            if plot_qshs_data:
                fig, axes = plt.subplots(2, 2, figsize=(19, 9))

        if plot_qshs_data:
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
            plt.savefig(f"{run_dir}/qshs_data_{spectra_list.index(s)}.png", dpi=150)
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

            plt.savefig(f"{run_dir}/smith_chart_{spectra_list.index(s)}.png", dpi=150)
            plt.close(fig)

print(f"{len(invalid_files)}/{ len(spectra_list)} spectra are empty or invalid.")
invalid_files_df = pd.DataFrame(invalid_files, columns=["bad_files"])
invalid_files_df.to_csv(f"{run_dir}/invalid_files.csv", index=False)
valid_files_df = pd.DataFrame(valid_files, columns=["valid_files"])
valid_files_df.to_csv(f"{run_dir}/valid_files.csv", index=False)
