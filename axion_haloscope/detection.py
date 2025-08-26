
# axion_haloscope/detection.py
from __future__ import annotations
import numpy as np
from scipy.stats import norm

def threshold_for_detection(target_snr: float = 5.0, confidence: float = 0.95) -> float:
    """
    Theta such that P[N(target_snr,1) > Theta] = confidence  ⇒  Theta = target_snr - Φ^{-1}(confidence).
    """
    return target_snr - norm.ppf(confidence)

def find_candidates(
    Dg: np.ndarray, sg: np.ndarray, thresh: float, min_separation: int = 3
):
    """
    Return (indices, z) where z = Dg/sg and z >= thresh, merging nearby peaks.
    """
    z = np.zeros_like(Dg)
    m = np.isfinite(sg) & (sg > 0)
    z[m] = Dg[m] / sg[m]
    raw = np.where(z >= thresh)[0].tolist()
    if not raw:
        return [], z
    merged = []
    i = 0
    while i < len(raw):
        j = i
        group = [raw[i]]
        while j + 1 < len(raw) and (raw[j+1] - raw[j]) <= min_separation:
            j += 1
            group.append(raw[j])
        merged.append(max(group, key=lambda k: z[k]))
        i = j + 1
    return merged, z
