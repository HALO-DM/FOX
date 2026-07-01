import h5py
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import sys, datetime, os

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


out_root = "output/qshs_import"
try:
    os.mkdir(out_root)
except FileExistsError:
    pass

timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
run_dir = f"{out_root}/run_{timestamp}"
os.mkdir(run_dir)
#run_dir.mkdir(parents=True, exist_ok=True)
for i in range(11):
    with h5py.File(f"input/Jan_QSHS/QSHS_2026-01-27_001{83+i}.hdf5", "r") as f:
        #f.visititems(print_hdf5_structure)
        #f.visititems(inspect)
        #data = hdf5_to_object(f)
        #print(data.attrs)
        power_spectra   = f["Power_Spectra"][()]
        raw_data        = f["Raw_Data"][()]
        narrow_vna_scan = f["Narrow_VNA_Scan"][()]
        wide_vna_scan   = f["Wide_VNA_Scan"][()]

        print (power_spectra[0])
        print (power_spectra[1])

        print (raw_data[0])         #empty
        print (raw_data[1])         #empty

        print (narrow_vna_scan[0])
        print (narrow_vna_scan[1])
        print (narrow_vna_scan[2])       

        print (wide_vna_scan[0])    #empty
        print (wide_vna_scan[1])    #empty
        print (wide_vna_scan[2])    #empty

        fs = 1_000_000  # Hz, replace this
        freq = np.linspace(0, fs / 2, power_spectra.shape[1])

        
        fig, axes = plt.subplots(2, 2, figsize=(14, 5))

        axes[0, 0].plot(raw_data[0],raw_data[1], label="Raw")
        axes[0, 0].set_title("Raw Data")
        axes[0, 0].set_xlabel("Sample index")
        axes[0, 0].set_ylabel("ADC value")
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        axes[1, 0].plot(power_spectra[0],power_spectra[1], label="Power Spectra")
        axes[1, 0].set_title("Power Spectra")
        axes[1, 0].set_xlabel("Frequency bin")
        axes[1, 0].set_ylabel("Power")
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        
        a = axes[0, 1].scatter(narrow_vna_scan[1],narrow_vna_scan[2], c=narrow_vna_scan[0], cmap='viridis', s=100, label="Narrow VNA Scan")
        axes[0, 1].set_title("Narrow VNA Scan")
        axes[0, 1].set_xlabel("Unknown")
        axes[0, 1].set_ylabel("Unknown")
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        fig.colorbar(a, ax=axes[0, 1])

        b = axes[1, 1].scatter(wide_vna_scan[1],wide_vna_scan[2], c=wide_vna_scan[0], cmap='viridis', s=100, label="Wide VNA Scan")
        axes[1, 1].set_title("Wide VNA Scan")
        axes[1, 1].set_xlabel("Unknown")
        axes[1, 1].set_ylabel("Unknown")
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        fig.colorbar(b, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f"{run_dir}/qshs_data_{i:03d}.png", dpi=150)
        plt.close(fig)


