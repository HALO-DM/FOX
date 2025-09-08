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

    1D speed PDF in Earth's frame (unit area).
    Uses analytic boosted MB (untruncated) with a practical truncation,
    and correctly falls back to the unboosted Maxwellian when v_earth≈0.
    Returns (v_grid, p(v)) with ∫ p(v) dv = 1.
    """
    v0 = float(v0); v_esc = float(v_esc); ve = float(v_earth)
    v_max = max(3.5*v0 + max(ve, 0.0), v_esc + max(ve, 0.0))
    v_grid = np.linspace(0.0, v_max, int(nv))

    # Unboosted isotropic Maxwellian (speed PDF) ∝ v^2 exp(-v^2/v0^2)
    def unboosted(v):
        p = (v**2) * np.exp(-(v*v)/(v0*v0 + 1e-300))
        p[v > v_esc] = 0.0
        return p

    if ve <= 1e-6:
        p = unboosted(v_grid)
    else:
        # Boosted MB speed PDF (untruncated analytic form) ∝ v * sinh(2 v ve / v0^2) * exp(-(v^2 + ve^2)/v0^2)
        x = 2.0 * v_grid * ve / (v0*v0 + 1e-300)
        p = v_grid * np.sinh(x) * np.exp(-(v_grid*v_grid + ve*ve) / (v0*v0 + 1e-300))
        # Practical truncation: no contributions when underlying halo speed would exceed v_esc
        # A conservative cutoff is v > v_esc + ve in the boosted frame.
        p[v_grid > (v_esc + ve)] = 0.0
        # If numerical underflow produced all-zeros (very extreme params), fall back to unboosted
        if not np.any(p > 0):
            p = unboosted(v_grid)

    p[p < 0] = 0.0
    norm = np.trapezoid(p, v_grid)
    if norm > 0:
        p /= norm
    else:
        # final safety: normalize unboosted form
        p = unboosted(v_grid)
        p /= max(np.trapz(p, v_grid), 1.0)
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
