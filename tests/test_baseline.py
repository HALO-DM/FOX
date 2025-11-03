# tests/test_remove_baseline.py
import numpy as np
import matplotlib
# Use a non-interactive backend for tests
matplotlib.use("Agg")
from matplotlib.figure import Figure
from axion_haloscope.baseline import remove_baseline


def _make_test_spectrum(n=501):
  """
  Produce a synthetic spectrum with a smooth baseline + a narrow peak.
  n should be odd and >= 401 to be compatible with the default SG window.
  """
  x = np.linspace(0.0, 10.0, n)
  baseline = 1.0 + 0.05 * x + 0.01 * x**2
  # single Gaussian peak centered near middle
  peak = 0.6 * np.exp(-0.5 * ((x - 5.0) / 0.08) ** 2)
  spectrum = baseline * (1.0 + peak)
  freqs_hz = np.linspace(1e9, 11e9, n)  # for testing frequency label handling
  return spectrum.astype(float), baseline.astype(float), freqs_hz


def test_basic_processing_reconstructs_spectrum():
  spectrum, true_baseline, _ = _make_test_spectrum()
  processed, baseline = remove_baseline(spectrum)

  # shapes preserved
  assert processed.shape == spectrum.shape
  assert baseline.shape == spectrum.shape

  # processed * baseline ≈ original spectrum
  np.testing.assert_allclose(processed * baseline, spectrum, rtol=1e-6, atol=1e-9)


def test_subtract_one_roundtrip():
  spectrum, _true_baseline, _ = _make_test_spectrum()
  processed, baseline = remove_baseline(spectrum, subtract_one=True)

  # After subtracting one, (processed + 1) * baseline should reconstruct the spectrum
  np.testing.assert_allclose((processed + 1.0) * baseline, spectrum, rtol=1e-6, atol=1e-9)


def test_diagnostic_true_returns_figure_and_axis_label():
  spectrum, _true_baseline, freqs_hz = _make_test_spectrum()
  processed, baseline, fig = remove_baseline(
    spectrum,
    diagnostic=True,
    freqs_hz=freqs_hz,
  )

  # types and shapes
  assert processed.shape == spectrum.shape
  assert baseline.shape == spectrum.shape
  assert isinstance(fig, Figure)

  # The bottom axis xlabel should be "Frequency [GHz]" when freqs_hz is provided
  # The figure has two axes (top and bottom); bottom is the second axis
  bottom_ax = fig.axes[1]
  assert bottom_ax.get_xlabel() == "Frequency [GHz]"

  # The top axis title should be the default title
  top_ax = fig.axes[0]
  assert top_ax.get_title() == "Baseline removal diagnostic"

  # close figure to avoid resource leak in test runner
  matplotlib.pyplot.close(fig)


def test_diagnostic_outfile_saves_and_returns_two_tuple(tmp_path):
  spectrum, _true_baseline, _ = _make_test_spectrum()
  outfile = tmp_path / "baseline_diag.png"

  processed, baseline = remove_baseline(
    spectrum,
    diagnostic={"outfile": str(outfile)},
  )

  # function should return a two-tuple (figure is saved and closed internally)
  assert processed.shape == spectrum.shape
  assert baseline.shape == spectrum.shape

  # file should exist and be non-empty
  assert outfile.exists()
  assert outfile.stat().st_size > 0