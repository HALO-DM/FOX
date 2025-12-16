# Creating a self-contained Python example that:
# - builds a synthetic "grand spectrum" (frequency bins with noise)
# - injects a clear candidate signal (Gaussian-shaped)
# - computes matched-filter amplitude estimator and 95% one-sided upper limits across the spectrum
# - converts amplitude limits to a coupling-like parameter g = sqrt(A) (simple model)
# - plots the grand spectrum (showing the injected candidate) and the exclusion plot (g_UL vs frequency)
# The plotting follows the python_user_visible tool rules: use matplotlib (no seaborn), one plot per figure,
# and do not specify colors or styles explicitly.
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- synthetic spectrum parameters ---
n_bins = 2000
freqs = np.linspace(1000.0, 2000.0, n_bins)  # arbitrary units (e.g., MHz)
df = freqs[1] - freqs[0]

# noise per bin (rms). Slightly varying to be a bit more realistic.
sigma = 1.0 + 0.05 * np.sin(2 * np.pi * freqs / 8000.0)
# ensure positive
sigma = np.abs(sigma)

# --- injected candidate ---
f0_inj = 1425.37  # injection center frequency
width_hz = 1.0    # Gaussian sigma in same units as freqs
# build Gaussian template (not normalized to unit area; treat A as amplitude multiplier)
def gaussian_template(center, width, freqs):
  return np.exp(-0.5 * ((freqs - center) / width)**2)

T_inj = gaussian_template(f0_inj, width_hz, freqs)

# compute matched-filter variance at injection location (to choose a visible amplitude)
S_inj = np.sum((T_inj**2) / (sigma**2))
var_A_inj = 1.0 / S_inj
std_A_inj = np.sqrt(var_A_inj)

# choose an injected amplitude that is clearly visible: e.g., 6-sigma detection
A_inj = 6.0 * std_A_inj

# build noisy grand spectrum and add injection
noise = np.random.normal(0.0, sigma)
G = noise + A_inj * T_inj

# --- sliding matched filter to produce estimator and variance at each candidate center ---
width_search = width_hz  # use same template width in search
A_hat = np.zeros_like(freqs)
var_A = np.zeros_like(freqs)

# to be computationally efficient, we'll precompute a template matrix window
# but for clarity (and since n_bins=2000) we'll loop straightforwardly
for i, fc in enumerate(freqs):
  T = gaussian_template(fc, width_search, freqs)
  denom = np.sum((T**2) / (sigma**2))
  num = np.sum((G * T) / (sigma**2))
  if denom <= 0:
    A_hat[i] = 0.0
    var_A[i] = np.inf
  else:
    A_hat[i] = num / denom
    var_A[i] = 1.0 / denom

# --- one-sided 95% upper limits on amplitude A ---
z_95 = 1.645  # 95% one-sided Gaussian quantile
A_UL = A_hat + z_95 * np.sqrt(var_A)

# physical amplitude is non-negative; clip the UL at zero lower bound before mapping to g
A_UL_clipped = np.maximum(A_UL, 0.0)

# simple physics mapping: assume measured power A ∝ g^2, so g = sqrt(A) (unit-normalized)
g_UL = np.sqrt(A_UL_clipped)

# also compute local SNR = A_hat / sqrt(var_A)
snr = A_hat / np.sqrt(var_A)

# --- plots ---

# 1) Grand spectrum (data) with injection annotated
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(freqs, G, label='Grand spectrum (data)')
# overlay the injected template scaled by injected amplitude to show its shape
ax1.plot(freqs, A_inj * T_inj, label='Injected signal (A_inj * template)')
ax1.set_xlabel('Frequency (arb. units)')
ax1.set_ylabel('Amplitude (arb. units)')
ax1.set_title('Synthetic grand spectrum with an injected candidate')
ax1.legend(loc='upper right')

# 2) Exclusion plot: g_UL vs frequency. Use log y-axis because limits vary widely.
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(freqs, g_UL, label='95% one-sided upper limit on g')
ax2.set_xlabel('Frequency (arb. units)')
ax2.set_ylabel('g_UL = sqrt(A_UL) (arb. units)')
ax2.set_title('Exclusion plot (95% one-sided)')
ax2.set_yscale('log')
ax2.axvline(f0_inj, linestyle='--', label='Injected frequency')
ax2.legend(loc='upper right')

# 3) Optional: SNR vs frequency to show where the candidate stands out
fig3, ax3 = plt.subplots(figsize=(9, 3))
ax3.plot(freqs, snr, label='Local matched-filter SNR')
ax3.set_xlabel('Frequency (arb. units)')
ax3.set_ylabel('SNR (A_hat / sigma_A)')
ax3.set_title('Local matched-filter SNR (shows the candidate peak)')
ax3.legend(loc='upper right')

# show the figures
plt.show()

# print a short numeric summary of the injection region
idx_peak = np.argmax(snr)
summary = {
  'injected_frequency': f0_inj,
  'found_peak_frequency': float(freqs[idx_peak]),
  'A_inj': float(A_inj),
  'A_hat_at_peak': float(A_hat[idx_peak]),
  'SNR_at_peak': float(snr[idx_peak]),
  'g_UL_at_peak': float(g_UL[idx_peak])
}
print('Summary of injection and recovered peak:')
for k, v in summary.items():
  print(f'  {k}: {v}')

