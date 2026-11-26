r"""
Camera-ready FLOPs plot (Figure 10 of the EACL paper).

Drawn at the printed size of one ACL column (3.03 in), so the fonts are the real
sizes on the page: 8 pt axis labels, 7 pt ticks, legend and annotations.
Include with \includegraphics[width=\linewidth]{FLOPS.png} (not width=0.9\linewidth
of a larger figure, which would shrink the fonts again).
Data values are unchanged from the original script.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------------------------------------------------------ style ---
COL_W, FIG_H = 3.03, 1.75   # inches; one ACL column

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ------------------------------------------------------------------- data ---
# token_rate = 10 * (1 - prune_rate/100)
prune_pct   = [0, 16.67, 28.57, 37.5, 44.44, 50, 66.67, 75, 82.77]
token_rate  = [10 * (1 - p / 100) for p in prune_pct]
total_flops = [16538.6, 13793.4, 11887.6, 10426.4, 9295.2, 8382.9, 5717.7, 4395.0, 3184.1]
x, y = np.array(token_rate), np.array(total_flops)

GRID, COLOR = "#cccccc", "#e05c00"

# ----------------------------------------------------------------- figure ---
fig, ax = plt.subplots(figsize=(COL_W, FIG_H))
ax.grid(True, color=GRID, linewidth=0.4, linestyle="--", zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor("#aaaaaa")
    spine.set_linewidth(0.6)

ax.plot(x, y, color=COLOR, linewidth=1.3, marker="o", markersize=4,
        markerfacecolor=COLOR, markeredgecolor="white", markeredgewidth=0.8,
        zorder=3, label="Total FLOPs")

box = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR, linewidth=0.7)
# first point (no pruning)
ax.annotate(f"{y[0]:,.0f}G", xy=(x[0], y[0]), xytext=(x[0] - 1.25, y[0] + 1500),
            fontsize=7, fontweight="bold", color=COLOR, bbox=box)
# last point (full blank removal)
ax.annotate(f"{y[-1]:,.0f}G\n(5.2x ↓)", xy=(x[-1], y[-1]),
            xytext=(x[-1] - 0.75, y[-1] + 3600),
            fontsize=7, fontweight="bold", color=COLOR, bbox=box,
            arrowprops=dict(arrowstyle="->", color=COLOR, lw=0.7,
                            connectionstyle="arc3,rad=-0.2"))

ax.set_xlim(0, 11)
ax.set_ylim(0, 20000)
ax.set_xticks(range(1, 11))
ax.set_yticks([0, 5000, 10000, 15000, 20000])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.tick_params(axis="both", colors="#444444", length=2, pad=1.5)
ax.set_xlabel("Token Rate (TPS)", fontweight="bold", labelpad=2)
ax.set_ylabel("GFLOPs", fontweight="bold", labelpad=2)
ax.legend(frameon=True, facecolor="white", edgecolor=GRID, loc="upper left",
          handlelength=1.6, borderpad=0.4)

fig.savefig("FLOPS.png", dpi=600)
fig.savefig("FLOPS.pdf")
print("done")
