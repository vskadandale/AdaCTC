r"""
Camera-ready mismatch heatmaps (Figures 8 and 9 of the EACL paper).

Drawn at the printed size of one ACL column (3.03 in), so the fonts below are
the real sizes on the page: 8 pt title/axis labels, 7 pt ticks, 6 pt cell values.
Include with \includegraphics[width=\linewidth]{ADA_WER.png} (not width=7cm).

Data values and the colour transformation are unchanged from the original script.
Cell values are shown with one decimal so that they fit the cells at a readable
size; set CELL_FMT = ".2f" (and use a two-column figure*) to keep two decimals.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------ style ---
COL_W, FIG_H = 3.03, 2.75     # inches; one ACL column
CELL_FMT = ".1f"              # ".2f" only fits if the figure is made wider
CELL_SIZE = 6                 # pt, numbers inside the cells
ROW_LABEL = "ADAPTIVE"        # matches \textsc{Adaptive} in the paper text and captions

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def mismatch_heatmap(rates, matrix_data, adaptive_data, transform, cbar_label_fn,
                     title, fname):
    df_matrix = pd.DataFrame(matrix_data, index=rates, columns=rates)
    df_adapt = pd.DataFrame(adaptive_data, index=[ROW_LABEL], columns=rates)
    m_trans = transform(df_matrix).fillna(0.0)
    a_trans = transform(df_adapt)

    fig, axs = plt.subplots(
        2, 2, figsize=(COL_W, FIG_H),
        gridspec_kw={"width_ratios": [15, 0.8], "height_ratios": [9, 1],
                     "wspace": 0.05, "hspace": 0.32},
    )
    ax1, cbar_ax, ax2 = axs[0, 0], axs[0, 1], axs[1, 0]
    axs[1, 1].axis("off")

    common = dict(fmt=CELL_FMT, cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0.3,
                  annot_kws={"size": CELL_SIZE})
    sns.heatmap(m_trans, annot=df_matrix, ax=ax1, cbar_ax=cbar_ax, **common)
    sns.heatmap(a_trans, annot=df_adapt, ax=ax2, cbar=False, **common)

    ax1.set_title(title, fontweight="bold", pad=4)
    ax1.set_ylabel("Training Token Rate (TPS)", fontweight="bold")
    ax2.set_xlabel("Inference Token Rate (TPS)", fontweight="bold")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=0)
    ax1.tick_params(axis="y", rotation=0)
    ax2.tick_params(axis="x", rotation=0)
    ax2.set_yticklabels([ROW_LABEL], va="center", rotation=0,
                        fontweight="bold", fontsize=5.5)

    cbar_ticks = [1.0, 0.75, 0.5, 0.25, 0.05]
    cbar_ax.set_yticks(cbar_ticks)
    cbar_ax.set_yticklabels([cbar_label_fn(y) for y in cbar_ticks])
    cbar_ax.tick_params(labelsize=7)

    fig.savefig(fname + ".png", dpi=600)
    fig.savefig(fname + ".pdf")
    plt.close(fig)


# ------------------------------------------------- WER (ASR, VoxPopuli) ------
rates = ["10", "8.34", "7.14", "6.25", "5.56", "5", "3.34", "2.5", "1.72"]
wer = [
    [8.63, 29.95, 31.02, 45.87, 70.66, 99.11, 115.32, 140.89, 212.5],
    [8.7, 8.33, 8.47, 9.99, 12.21, 18.18, 85.45, 96.18, 165.01],
    [9.42, 8.47, 8.53, 8.51, 9.55, 10.88, 26.77, 44.34, 86.18],
    [9.11, 8.56, 8.86, 8.48, 8.44, 8.7, 16.19, 207.33, 503.57],
    [8.98, 8.71, 8.64, 8.5, 8.58, 8.84, 9.54, 22.28, 65.68],
    [11.0, 9.75, 8.8, 8.65, 8.65, 8.92, 9.72, 18.05, 46.21],
    [22.66, 9.87, 9.27, 9.18, 9.08, 9.01, 8.74, 11.06, 35.87],
    [105.48, 56.46, 28.93, 18.34, 12.44, 10.17, 8.89, 9.82, 16.36],
    [107.82, 255.16, 161.25, 85.96, 53.88, 43.16, 26.32, 20.79, 12.33],
]
wer_adaptive = [[8.24, 8.28, 8.16, 8.26, 8.27, 8.36, 8.26, 8.69, 10.91]]
vmin_val, decay = 8.16, 3.0
mismatch_heatmap(
    rates, wer, wer_adaptive,
    transform=lambda df: np.exp(-(df - vmin_val) / decay),
    cbar_label_fn=lambda y: f"{vmin_val - decay * np.log(y):.1f}" if y > 0 else ">20",
    title=r"WER ($\downarrow$) in mismatched conditions",
    fname="ADA_WER",
)

# ------------------------------------------- BLEU (ST, EuroParl-ST) ----------
rates = ["10", "8.34", "7.14", "6.25", "5.56", "5", "3.34", "2.5", "1.49"]
bleu = [
    [27.72, 24.49, 12.94, 6.44, 4.69, 4.15, 2.23, 1.14, 0.04],
    [29.00, 29.67, 29.17, 27.22, 22.84, 17.72, 5.58, 8.21, 3.35],
    [28.67, 28.87, 29.23, 29.07, 28.59, 28.22, 17.93, 10.08, 2.56],
    [27.55, 28.48, 28.76, 28.85, 29.03, 28.59, 27.70, 23.25, 6.18],
    [25.43, 27.02, 27.15, 27.23, 27.29, 27.04, 26.75, 24.57, 9.98],
    [27.89, 28.22, 28.91, 29.15, 29.42, 29.68, 28.92, 26.80, 8.26],
    [24.90, 25.31, 26.21, 26.79, 27.15, 27.46, 28.31, 25.95, 4.44],
    [5.89, 17.01, 21.25, 24.45, 25.48, 26.30, 27.13, 27.05, 16.14],
    [8.38, 11.36, 12.85, 14.37, 14.79, 15.84, 17.95, 19.84, 21.62],
]
bleu_adaptive = [[30.34, 30.35, 30.26, 30.35, 30.30, 30.31, 30.02, 29.55, 21.33]]
vmax_val = 30.35
mismatch_heatmap(
    rates, bleu, bleu_adaptive,
    transform=lambda df: np.exp(-(vmax_val - df) / decay),
    cbar_label_fn=lambda y: f"{vmax_val + decay * np.log(y):.1f}" if y > 0 else "<20",
    title=r"Normalized BLEU ($\uparrow$) in mismatched conditions",
    fname="ADA_BLEU",
)
print("done")
