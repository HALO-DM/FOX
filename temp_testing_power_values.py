import matplotlib.pyplot as plt
import numpy as np
import csv

from matplotlib.ticker import PercentFormatter

columns = {}

with open("power.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)

# Each column is now an array (list)
power_values = np.array(columns["power"], dtype=float)
power_array = np.array(columns["success"], dtype=float)


y_vals = (power_array / 30) * 100


# visual style (small rc tweaks)
plt.rcParams.update({
    'figure.figsize': (9, 3),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.6,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
})

threshold = 95.0  # percent threshold
plt.plot(power_values, y_vals, lw=1.8, color='#1f77b4', zorder=2)
plt.scatter(
    power_values, y_vals,
    marker='x', s=36, linewidths=1.2,
    edgecolors='#0b3d91', facecolors='none', zorder=3
)

# horizontal threshold line (95%)
plt.axhline(threshold, color="#666666", linestyle='--', linewidth=1.2, zorder=1)
plt.text(
    power_values.min() + 5, threshold - 10,
    s = f'Confidence Level: {threshold:.0f}%',
    va='bottom', ha='right', fontsize=9, color="#444444"
)

# find first crossing: y >= threshold
cross_idx = np.where(y_vals >= threshold)[0]
if cross_idx.size == 0:
    # no crossing found: annotate that fact
    plt.text(
        0.98, 0.02,
        'No crossing of 95% within given range',
        transform=plt.gca().transAxes,
        ha='right', va='bottom',
        fontsize=9, color='#a00'
)
else:
# take first crossing and interpolate for a more precise x
    i1 = int(cross_idx[0])
    if i1 == 0:
        x_cross = float(power_values[0])
    else:
        x0, y0 = float(power_values[i1-1]), float(y_vals[i1-1])
        x1, y1 = float(power_values[i1]), float(y_vals[i1])
        # avoid division by zero; if y1==y0 fallback to x1
        if y1 == y0:
            x_cross = x1
        else:
            t = (threshold - y0) / (y1 - y0)
            x_cross = x0 + t * (x1 - x0)

    # vertical line at the interpolated x
    plt.axvline(x_cross, color='#d62728', linestyle='--', linewidth=1.2, zorder=1)

    # mark intersection point and annotate x value
    plt.scatter([x_cross], [threshold], s=80, facecolors='white',
                edgecolors='#d62728', linewidths=1.6, zorder=5)
    plt.annotate(
        f'x = {x_cross:.3g}',
        xy=(x_cross + 0.5, threshold- 25),
        xytext=(10, 18),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', lw=0.8),
        fontsize=9,
        ha='left', va='bottom',
        color='#333333'
    )

# labels, title, percent formatter
plt.xlabel("Power Values [arbitrary]")
plt.ylabel("Recovery Rate (%)")
plt.title("Axion Recovery Rate")
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
plt.grid(which='major', alpha=0.25)

plt.tight_layout()
plt.savefig("Power Values.png", dpi=500, bbox_inches='tight')
plt.close()