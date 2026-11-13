#python scripts/data_analysis/tagging_midas_yaml.py configs/midas_tagging.yaml
from __future__ import annotations
import argparse
import matplotlib
import h5py
import numpy as np
import pandas as pd
import datetime, os, sys
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
import matplotlib.cm as cm
from tqdm import tqdm
import psycopg2

from axion_haloscope.file_converter import MidasFileReader

from axion_haloscope.data_quality_working import power_too_high, small_bandwidth, metadata_is_zeros, too_noisy
from axion_haloscope.utils                import create_directory, find_project_root, load_yaml_config

# --------------------------
# Read Midas to sset
#---------------------------

ap = argparse.ArgumentParser(description="Simulate haloscope run from YAML config")
ap.add_argument("config", help="Path to YAML config (e.g. configs/simulate_run.yaml)")
args = ap.parse_args()

cfg_path = pathlib.Path(args.config).resolve()
if not cfg_path.exists():
    sys.exit(f"Config file not found: {cfg_path}")
cfg = load_yaml_config(cfg_path)
inp, tag, sql, sim, qc, alg, base, rb, det, out, diag = (cfg[k] for k in ("input", "tagging", "sql", "simulation", "quality", "alignment","baseline","rebin","detection","output","diagnostic"))
project_root = find_project_root(pathlib.Path(__file__).resolve())
out_root = project_root / out["root"] / "data_analysis"
timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
run_dir = create_directory(out_root, f'{out["subdir_prefix"]}_{timestamp}')

main_plots_dir = create_directory(run_dir, "main_plots")
data_dir = create_directory(run_dir, "data")
                                    

directory = Path(inp["directory"])
files = sorted(directory.glob("*.mid.lz4"))

if not files:
    raise FileNotFoundError(f"No Midas files in {directory}")

# make a directory for bit files if necessary
if tag["save_error_files"] == True:
    os.mkdir(f"output/error_files/error_files_{timestamp}")
    os.mkdir(f"output/error_files/bit_files_{timestamp}")

#redefining power too high, too noisy to not require metadata
def power_too_high(
    s: np.ndarray,
    f: np.ndarray,
    i: int,
    *,
    p_max: float = 1e-50,
) -> bool:
    average_power = np.average(s)

    return average_power > p_max

def too_noisy(
    s: np.ndarray,
    f: np.ndarray,
    i: int,
    *,
    rms_max: float = 100,
    nan_fail: bool = True,
    robust: bool = True,
) -> bool:
    if nan_fail and (not np.isfinite(s).all()):
        return True
    x = s
    if robust:
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))  # ≈ 0.6745 σ for Gaussian
        sigma = mad / 0.6744897501960817 if mad > 0 else np.nanstd(x)
        rms = np.sqrt(np.nanmean((x - med) ** 2))
    else:
        mu = np.nanmean(x)
        sigma = np.nanstd(x)
        rms = np.sqrt(np.nanmean((x - mu) ** 2))
    if not np.isfinite(sigma):  # degenerate edge case
        return True

    return rms > rms_max



cavity_in_chain = tag["cavity"]

md_tot=np.zeros(10)
cav_md_tot=np.zeros(5)
freqs_per_spec=[]
spectra=[]
bits_upload=[]
ii=[]
bits_tot=[]
for i in tqdm(range(len(files)), "Processing Midas files"):
    FILENAME=str(files[i])
    reader = MidasFileReader(FILENAME, prescale=1)
    sset = reader.read_midas_as_sset()
    freqs_per_spec.append(sset.freqs_per_spec[i])
    spectra.append(sset.spectra[i])
    # -----------------------------------
    # Missing/Invalid metadata
    # -----------------------------------
    missing_metadata=int(not sset.metadata)
    cav_bits=[] #cav in chain, missing all, b_vals,q_factors,res_freqs,cw_freqs,bandwidths
    metadata_bits=[] #type,x,y,sample_rate,start,stop,start_binary,stop_binary,n_run#
    meta_invalid=[] #x,y,invalid_sample_rate,sample_time_mismatch
    power_spec_bits=[] # too noisy, power too high
    cav_bits.append(int(cavity_in_chain))
    if missing_metadata == 0:
        meta=sset.metadata
        metadata_bits.append(0)
        metadata_bits.append(int("type" not in meta))
        if "x_unit" not in meta:
            metadata_bits.append(1)
            meta_invalid.append(1)
        else:
            metadata_bits.append(0)
            meta_invalid.append(int(meta["x_unit"] not in ["Hz","MHz","GHz"]))
        if "y_unit" not in meta:
            metadata_bits.append(1)
            meta_invalid.append(1)
        else:
            metadata_bits.append(0)
            meta_invalid.append(int(meta["y_unit"] not in ["V^2/Hz"]))
        if "sample_rate" not in meta:
            metadata_bits.append(1)
            meta_invalid.append(1)
        else:
            metadata_bits.append(0)
            meta_invalid.append(int(meta["sample_rate"]<2.5e6))
        metadata_bits.append(int("start_time" not in meta))
        d1=datetime.datetime.strptime(meta["start_time"],'%a %b %d %H:%M:%S %Y')
        d2=datetime.datetime.strptime(meta["stop_time"],'%a %b %d %H:%M:%S %Y')
        meta_invalid.append(int(d1>d2))
        metadata_bits.append(int("stop_time" not in meta))
        metadata_bits.append(int("start_time_binary" not in meta))
        metadata_bits.append(int("stop_time_binary" not in meta))
        metadata_bits.append(int("n_run" not in meta))
        if cavity_in_chain:
            missing_cav_md=int("cavity_metadata" not in meta)
            cav_bits.append(missing_cav_md)
            if missing_cav_md==0:
                cavity_md=meta.cavity_metadata
                cav_bits.append(int("b_vals" not in cavity_metadata))
                cav_bits.append(int("q_factors" not in cavity_metadata))
                cav_bits.append(int("res_freqs" not in cavity_metadata))
                cav_bits.append(int("cw_freqs" not in cavity_metadata))
                cav_bits.append(int("bandwidths" not in cavity_metadata))
            else:
                for m in range (5):
                    cav_bits.append(1)
        else:
            for m in range(6):
                cav_bits.append(0)
    else:
        metadata_bits=np.ones(10)
        if cavity_in_chain:
            for m in range(6):
                cav_bits.append(1)
        else:
            for m in range(6):
                cav_bits.append(0)
    md_tot=md_tot+metadata_bits
    # ---------------------------------
    # Power spec quality
    # ---------------------------------
    power_spec_bits.append(int(too_noisy(sset.spectra[i],sset.freqs_per_spec[i],i,rms_max=qc["rms_max"],nan_fail=qc["nan_fail"],robust=qc["robust"])))
    power_spec_bits.append(int(power_too_high(sset.spectra[i],sset.freqs_per_spec[i],i,p_max=qc["p_max"])))
    bits=metadata_bits+cav_bits+meta_invalid+power_spec_bits
    bits_tot.append(bits)
    bits=np.pad(bits,(64-len(bits), 0),'constant',constant_values=(0,0))
    bits="".join(str(x) for x in (bits))
    run_no=[17,18,19,6] #change depending on how runs are labelled
    bitss=(str(run_no[i]),bits)
    bits_upload.append(bitss)
    if tag["save_error_files"] == True:
        with open(f"output/error_files/bit_files_{timestamp}/{i}","w") as fi:
                bit_str=[str(item) for item in bits]
                writedata='\n'.join(bit_str)
                fi.writelines(writedata)


    # --------------------------------------
    # Uploading to db
    # --------------------------------------

if tag["upload_to_db"] == True:
    host = sql["host"]
    user = sql["user"]
    port = sql["port"]
    db = sql["db"]
    psqlConnect = "dbname=%s user=%s host=%s port=%d" % (db,user,host,port)
    try:
        db = psycopg2.connect(psqlConnect)
        schema = 'private'
        DBconn = db.cursor()
    except:
        print("SQL error ...")
    sql = "INSERT INTO run_status (run_number, status) VALUES (%s, %s)"
    val = bits_upload
    DBconn.executemany(sql,val)

    db.commit()
    print(DBconn.rowcount, "run statuses inserted in axion_db")

if tag["diagnostic_mode"] == True:
    sumindex = [sum(elts) for elts in zip(*bits_tot)]
    mam,mt,mx,my,msr,mstar,mstop,mstarb,mstopb,mn,cic,mc,bv,qf,rf,cf,bw,ix,iy,isr,ist,tn,pth = sumindex[0:23]
    print("-----------------------------------------")
    print("Missing metadata issues:")
    print("-----------------------------------------")
    bad_runs_md=[["Missing all metadata",mam],
    ["Missing type",mt],["Missing x unit", mx], ["Missing y unit", my],["Missing sample rate",msr],["Missing start",mstar],["Missing stop",mstop],["Missing start binary",mstarb],["Missing stop binary",mstopb],["Missing n run",mn]]
    df_md=pd.DataFrame(bad_runs_md, columns=["Issue", "No of spectra"])
    print(df_md)

    if cavity_in_chain == True:
        print("-----------------------------------------")
        print("Missing cavity metadata:")
        print("-----------------------------------------")
        bad_runs_cmd=[["Cavity in chain",cic],["Missing all cav md",mc],["Missing b val",bv],["Missing q factor",qf],["Missing res freq",rf],["Missing cw freq",cf],["Missing bandwidth",bw]]
        df_cmd=pd.DataFrame(bad_runs_cmd, columns=["Issue", "No of spectra"])
        print(df_cmd)
    print("-----------------------------------------")
    print("Invalid metadata issues:")
    print("-----------------------------------------")
    bad_runs_inv=[["Invalid x unit",ix],["Invalid y unit",iy],["Invalid sample rate",isr], ["Invalid stop time",ist]]
    df_inv=pd.DataFrame(bad_runs_inv, columns=["Issue", "No of spectra"])
    print(df_inv)

    print("-----------------------------------------")
    print("Bad Power Spectra:")
    print("-----------------------------------------")
    bad_runs_pwr=[["Too noisy",tn],["Power too high",pth]]
    df_pwr=pd.DataFrame(bad_runs_pwr, columns=["Issue", "No of spectra"])
    print(df_pwr)

    error_names = [
    "Missing all metadata", "Missing type", "Missing x unit",
    "Missing y unit", "Missing sample rate", "Missing start",
    "Missing stop", "Missing start binary", "Missing stop binary",
    "Missing n_run", "Cavity not in chain", "Missing cavity metadata", "Missing b_vals",
    "Missing q_factors", "Missing res_freqs", "Missing cw_freqs",
    "Missing bandwidths", "Invalid x unit", "Invalid y unit",
    "Invalid sample rate", "Invalid stop time", "Too noisy",
    "Power too high",
]


    error_df = pd.DataFrame(
        bits_tot,
        index=[file.name for file in files],
        columns=error_names,
    )

    fig, ax = plt.subplots(figsize=(14, max(5, len(files) * 0.3)))
    cmap =matplotlib.colors.ListedColormap(["white","blueviolet"])
    image = ax.imshow(error_df, aspect="auto", interpolation="nearest",
                      cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(len(error_df)))
    ax.set_yticklabels(error_df.index)
    ax.set_xticks(range(len(error_names)))
    ax.set_xticklabels(error_names, rotation=60)
    ax.set_xlabel("Error")
    ax.set_ylabel("File")
    fig.colorbar(image, ax=ax, label="Error present")
    fig.tight_layout()
    fig.savefig("errors_by_file.png", dpi=150)
    plt.close(fig)




