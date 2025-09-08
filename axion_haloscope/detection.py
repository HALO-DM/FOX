
# axion_haloscope/detection.py
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def threshold_for_detection(target_snr: float,
                            confidence: float,
                            n_trials: int = 1) -> float:
    """
    Return the z-threshold (σ units) so that the global false-alarm probability
    across n_trials is 1 - confidence.

    Parameters
    ----------
    target_snr : float
        (Kept for API compatibility; not used in the tail calculation.)
    confidence : float
        Desired global confidence (e.g. 0.95, 0.99).
    n_trials : int
        Number of independent trials (look-elsewhere factor).

    Returns
    -------
    theta : float
        Right-tail z cut. Monotone in 'confidence' and 'n_trials'.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1)")

    # Global → per-trial false-alarm rate
    alpha_global = 1.0 - confidence
    n_trials = max(1, int(n_trials))
    alpha_per = 1.0 - (1.0 - alpha_global)**(1.0 / n_trials)

    # Right-tail threshold: P(Z > theta) = alpha_per
    theta = norm.isf(alpha_per)
    return float(theta)


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
