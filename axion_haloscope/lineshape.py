# axion_haloscope/lineshape.py
"""
Lineshape
=========

Used to generate axion lineshape from cosmology.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

C = 299_792_458.0  # m/s


def shm_speed_pdf(
    v0=220e3,
    v_esc=544e3,
    v_earth=232e3,
    nv=20000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 1D speed PDF in Earth's frame for the Standard Halo Model.

    Parameters
    ==========
    v0 : float
        Dispersion (circular speed) [m/s]
    v_esc : float
        Galactic escape speed [m/s]
    v_earth : float
        Earth's speed through halo [m/s]
    nv : int
        Number of velocity samples

    Returns
    =======
    v_grid : ndarray
        Velocity grid [m/s]
    p_v : ndarray
        Probability density p(v), normalized so ∫ p(v) dv = 1

    1D speed PDF in Earth's frame (unit area).
    Uses analytic boosted MB (untruncated) with a practical truncation,
    and correctly falls back to the unboosted Maxwellian when v_earth≈0.
    Returns (v_grid, p(v)) with ∫ p(v) dv = 1.
    """
    v0 = float(v0)
    v_esc = float(v_esc)
    ve = float(v_earth)
    v_max = max(3.5*v0 + max(ve, 0.0), v_esc + max(ve, 0.0))
    v_grid = np.linspace(0.0, v_max, int(nv))

    def unboosted(v):
        """Unboosted isotropic Maxwellian (speed PDF) ∝ v^2 exp(-v^2/v0^2)"""
        p = (v**2) * np.exp(-(v*v)/(v0*v0 + 1e-300))
        p[v > v_esc] = 0.0
        return p

    if ve <= 1e-6:
        p = unboosted(v_grid)
    else:
        # Boosted MB speed PDF ∝ v * sinh(2 v ve / v0^2) * exp(-(v^2 + ve^2)/v0^2)
        # (untruncated analytic form)
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



def shm_maxwell_profile(
    freqs_hz: np.ndarray,
    f0_hz: float,
    v0=220e3,
    v_esc=544e3,
    v_earth=232e3
) -> np.ndarray:
    """
    Map SHM speed distribution to frequency-space power profile.

    Parameters
    ==========
    freqs_hz: np.ndarray
        Frequency axis of spectra
    f0_hz: float
        Axion/Resonant Frequency
    v0 : float
        Dispersion (circular speed) [m/s]
    v_esc : float
        Galactic escape speed [m/s]
    v_earth : float
        Earth's speed through halo [m/s]

    Returns
    =======
    prof: np.ndarray
        Power profile evaluated at `freqs_hz`, normalized so that `prof.sum() == 1` 
        (unit-sum over the input grid).
    """
    v_grid, p_v = shm_speed_pdf(v0=v0, v_esc=v_esc, v_earth=v_earth)
    f_shift = 0.5 * (f0_hz / (C*C)) * (v_grid**2)
    f_vals = f0_hz + f_shift
    prof = np.interp(freqs_hz, f_vals, p_v * v_grid, left=0.0, right=0.0)
    s = prof.sum()
    return prof / s if s > 0 else prof

def shm_maxwell_template(
    k:int,
    bin_width_hz: float,
    f0_hz: float,
    v0=220e3,
    v_esc=544e3,
    v_earth=232e3
) -> np.ndarray:
    """
    Build a k-bin template for matched filtering, centered at f0_hz.

    Parameters
    ==========
    k:int
        SHM maxwell template bin width
    bin_width_hz: float
        Width of a single bin
    f0_hz: float
        Axion/Resonant Frequency
    v0 : float
        Dispersion (circular speed) [m/s]
    v_esc : float
        Galactic escape speed [m/s]
    v_earth : float
        Earth's speed through halo [m/s]

    Returns
    =======
    t: np.ndarray
        k-bin template, normalized so `t.sum() == 1`
    """
    centers = np.arange(k) - (k-1)/2.0
    f_grid = f0_hz + centers * bin_width_hz
    t = shm_maxwell_profile(f_grid, f0_hz=f0_hz, v0=v0, v_esc=v_esc, v_earth=v_earth)
    return t / (t.sum() if t.sum() > 0 else 1.0)
