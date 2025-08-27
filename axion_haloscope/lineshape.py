# axion_haloscope/lineshape.py
from __future__ import annotations
import numpy as np

c = 299_792_458.0  # m/s

def _maxwell_3d(v, v0, v_esc):
    """Unboosted truncated Maxwellian (not yet 4π v^2 factor)."""
    f = np.exp(-v*v/(v0*v0))
    f[v >= v_esc] = 0.0
    return f

def shm_speed_pdf(v, v0=220e3, v_esc=544e3, v_earth=232e3, nv=20000):
    """
    1D speed PDF in Earth's frame (unit area), via numerical convolution (boost).
    Uses isotropic MB with Galilean boost by v_earth and truncation at v_esc.
    Returns (v_grid, p(v)) where integral p(v) dv = 1.
    """
    v_max = max(3.5*v0 + v_earth, v_esc + v_earth)
    v_grid = np.linspace(0.0, v_max, nv)
    # approximate boost by sampling angles analytically: p(v) ∝ v * sinh(2 v v_e / v0^2) * exp(-(v^2+v_e^2)/v0^2)
    # but we enforce truncation at v_esc by zeroing where original speed exceeds v_esc.
    ve = float(v_earth); v0=float(v0); ves=float(v_esc)
    x = 2.0 * v_grid * ve / (v0*v0 + 1e-30)
    p = v_grid * np.sinh(x) * np.exp(-(v_grid*v_grid + ve*ve)/(v0*v0 + 1e-30))
    # truncate: only count configurations with |u|<v_esc; approximate by zeroing for v_grid > (v_esc+ve)
    p[v_grid > (ves + ve)] = 0.0
    p[p<0]=0.0
    # normalize
    norm = np.trapz(p, v_grid)
    if norm <= 0: 
        p[:] = 0.0
    else:
        p /= norm
    return v_grid, p

def shm_maxwell_profile(freqs_hz: np.ndarray, f0_hz: float, v0=220e3, v_esc=544e3, v_earth=232e3) -> np.ndarray:
    """
    Power vs frequency for SHM. Map speed v to frequency shift Δf = f0 * v^2/(2 c^2).
    Profile ∝ p_speed(v) * (dv/df)^{-1} ∝ p_speed(v) * v,
    since df/dv = f0 * v / c^2.
    """
    # Build speed PDF
    v_grid, p_v = shm_speed_pdf(v0=v0, v_esc=v_esc, v_earth=v_earth)
    # Map v -> f
    df_dv = (f0_hz / (c*c)) * v_grid
    f_shift = 0.5 * (f0_hz / (c*c)) * (v_grid**2)  # Δf >= 0
    f_vals = f0_hz + f_shift
    # Jacobian: p(f) df = p(v) dv = p(v) (df/dv)^{-1} df  ⇒ p(f) ∝ p(v) * v
    p_f = p_v * v_grid
    # Bin p_f onto provided freqs_hz grid
    prof = np.interp(freqs_hz, f_vals, p_f, left=0.0, right=0.0)
    # Normalize to unit sum on the discrete grid
    s = prof.sum()
    return prof / s if s > 0 else prof

def shm_maxwell_template(K:int, bin_width_hz:float, f0_hz:float, v0=220e3, v_esc=544e3, v_earth=232e3) -> np.ndarray:
    """
    K-bin template centered at f0_hz on a grid with spacing bin_width_hz.
    """
    centers = np.arange(K) - (K-1)/2.0
    f_grid = f0_hz + centers * bin_width_hz
    T = shm_maxwell_profile(f_grid, f0_hz=f0_hz, v0=v0, v_esc=v_esc, v_earth=v_earth)
    S = T.sum()
    return T / S if S>0 else T
