# axion_haloscope/limit.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def compute_local_snr_template(sr: np.ndarray, Lq: np.ndarray) -> np.ndarray:
    """
    R_local(center) = sqrt( sum_i Lq[i]^2 / sr[i]^2 ) at each valid center (NaN at edges).
    """
    n, K = len(sr), len(Lq)
    R = np.full(n, np.nan)
    for r in range(n - K + 1):
        segs = sr[r:r+K]
        denom = (Lq*Lq / (segs**2 + 1e-18)).sum()
        if denom > 0:
            R[r + K//2] = np.sqrt(denom)
    return R

def coupling_limit(
    R_local: np.ndarray, target_snr: float = 5.0, g0: float = 1e-13, snr_efficiency: float = 0.90
) -> np.ndarray:
    """
    g_min(f) = g0 * sqrt( target_snr / (snr_efficiency * R_local) ).
    """
    out = np.full_like(R_local, np.nan, float)
    good = np.isfinite(R_local) & (R_local > 0)
    out[good] = g0 * np.sqrt(target_snr / (snr_efficiency * R_local[good]))
    return out

def plot_exclusion(freqs_r_hz, gmin, outfile=None, title="95% CL Exclusion (toy)"):
    plt.figure(figsize=(9,4))
    plt.plot(np.asarray(freqs_r_hz)/1e9, gmin, lw=1.5)
    plt.xlabel("Frequency [GHz]"); plt.ylabel(r"$g_{a\gamma\gamma}$ (arb vs $g_0$)")
    plt.title(title); plt.grid(alpha=0.3)
    if outfile:
        plt.tight_layout(); plt.savefig(outfile, dpi=160)
    return plt.gca()
