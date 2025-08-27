# axion_haloscope/lineshape.py
from __future__ import annotations
import numpy as np

c = 299_792_458.0  # m/s


def shm_speed_pdf(v0=220e3, v_esc=544e3, v_earth=232e3, nv=20000):
    """
    Generate 1D speed PDF in Earth's frame for the Standard Halo Model.
    Parameters
    ----------
    v0 : float
        Dispersion (circular speed) [m/s]
    v_esc : float
        Galactic escape speed [m/s]
    v_earth : float
        Earth's speed through halo [m/s]
    nv : int
        Number of velocity samples

    Returns
    -------
    v_grid : ndarray
        Velocity grid [m/s]
    p_v : ndarray
        Probability density p(v), normalized so ∫ p(v) dv = 1
    """
    v_max = max(3.5*v0 + v_earth, v_esc + v_earth)
    v_grid = np.linspace(0.0, v_max, nv)

    ve = float(v_earth); v0=float(v0); ves=float(v_esc)
    x = 2.0 * v_grid * ve / (v0*v0 + 1e-30)
    p = v_grid * np.sinh(x) * np.exp(-(v_grid*v_grid + ve*ve)/(v0*v0 + 1e-30))
    # truncate above escape speed
    p[v_grid > (ves + ve)] = 0.0
    p[p < 0] = 0.0
    # normalize
    norm = np.trapz(p, v_grid)
    p /= norm if norm > 0 else 1.0
    return v_grid, p

def shm_maxwell_profile(freqs_hz: np.ndarray, f0_hz: float, v0=220e3, v_esc=544e3, v_earth=232e3) -> np.ndarray:
    """
    Map SHM speed distribution to frequency-space power profile.
    """
    v_grid, p_v = shm_speed_pdf(v0=v0, v_esc=v_esc, v_earth=v_earth)
    f_shift = 0.5 * (f0_hz / (c*c)) * (v_grid**2)
    f_vals = f0_hz + f_shift
    prof = np.interp(freqs_hz, f_vals, p_v * v_grid, left=0.0, right=0.0)
    s = prof.sum()
    return prof / s if s > 0 else prof

def shm_maxwell_template(K:int, bin_width_hz:float, f0_hz:float, v0=220e3, v_esc=544e3, v_earth=232e3) -> np.ndarray:
    """
    Build a K-bin template for matched filtering, centered at f0_hz.
    """
    centers = np.arange(K) - (K-1)/2.0
    f_grid = f0_hz + centers * bin_width_hz
    T = shm_maxwell_profile(f_grid, f0_hz=f0_hz, v0=v0, v_esc=v_esc, v_earth=v_earth)
    return T / (T.sum() if T.sum() > 0 else 1.0)
