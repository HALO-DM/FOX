import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generate noisy data
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 100)
y_noisy = np.sin(x) + 0.2 * np.random.normal(size=len(x))

# Apply Savitzky-Golay filter
window_size = 7
order = 3
y_smoothed = savgol_filter(y_noisy, window_size, order)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_noisy, label="Noisy Data", marker="o")
plt.plot(x, y_smoothed, label="Smoothed Data", linestyle="--", linewidth=2)
plt.legend()
plt.title("Smoothing a Curve Using Savitzky-Golay Filter")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
