r"""
Camera-ready plots for the EACL paper.

Figures are drawn at their printed size (one ACL column = 3.03 in wide), so the
fonts below are the actual sizes in the PDF: 8 pt labels, 7 pt ticks/legend.
Include them in LaTeX with \includegraphics[width=\linewidth]{<name>.pdf}
(do NOT rescale with width=7cm, that would shrink the fonts again).
Data values are unchanged from the original script.
r"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ------------------------------------------------------------------ style ---
COL_W = 3.03          # ACL single-column width in inches
FIG_H = 2.3           # height in inches
SHOW_TITLE = True     # set False to drop the titles (the captions describe the plots)

plt.rcParams.update({
    "font.family": "serif",          # matches the Times body text
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,              # embed TrueType fonts (required by many venues)
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

STYLE = {  # name: (color, marker, linestyle)
    "CTC":     ("red",    "o", "-"),
    "COSD":    ("blue",   "x", "--"),
    "DS":      ("green",  "^", "-."),
    "Random":  ("orange", "s", ":"),
    "AvgPool": ("purple", "d", "-"),
}
ORDER = ["CTC", "COSD", "DS", "Random", "AvgPool"]


def new_fig():
    return plt.subplots(figsize=(COL_W, FIG_H))


def plot_methods(ax, data, **kw):
    for name in ORDER:
        if name in data:
            x, y = data[name]
            c, m, ls = STYLE[name]
            ax.plot(x, y, color=c, marker=m, linestyle=ls, label=name, **kw)


def finish(fig, ax, ylabel, title, fname, xticks=np.arange(0, 11, 1.0)):
    ax.set_xlabel("Token Rate (TPS)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.grid(True, linestyle="--", alpha=0.6)
    # one legend row above the axes: never covers data, replaces the title
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
              handlelength=1.6, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
    if SHOW_TITLE:
        ax.set_title(title, pad=17, fontweight="bold")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=600)
    plt.close(fig)


T = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]  # common token rates

# ---------------------------------------------- SER x IEMOCAP x FROZEN WHISPER
fig, ax = new_fig()
plot_methods(ax, {
    "CTC":     ([5, 5.56, 6.25, 7.143, 8.34, 10.0], [61.80, 59.46, 60.05, 62.28, 63.44, 68.74]),
    "COSD":    (T, [50.40, 55.31, 59.77, 62.07, 63.51, 61.57, 63.92, 65.48, 65.80, 68.74]),
    "DS":      (T, [53.69, 57.35, 57.86, 61.85, 65.50, 66.17, 67.79, 64.79, 64.96, 68.74]),
    "Random":  (T, [63.21, 64.66, 67.32, 67.58, 68.43, 69.45, 69.94, 70.56, 69.18, 70.11]),
    "AvgPool": (T, [67.37, 68.84, 68.01, 68.83, 70.42, 68.99, 67.66, 70.34, 68.55, 68.85]),
})
finish(fig, ax, "UWA (%)", r"%UWA ($\uparrow$) for ER with Whisper CTC (frozen backbone)",
       "SER_plots_frozen_backbone")

# -------------------------------------------- ASR x VoxPopuli x FT WHISPER (inset)
asr = {
    "CTC":     ([1.7229, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0],
                [12.33, 9.82, 8.74, 8.92, 8.58, 8.48, 8.53, 8.33, 8.63]),
    "COSD":    (T, [47.82, 22.27, 11.71, 10.03, 9.68, 8.61, 8.79, 8.93, 8.46, 8.63]),
    "DS":      (T, [62.53, 19.42, 11.23, 9.75, 10.16, 8.81, 8.82, 8.79, 8.73, 8.63]),
    "Random":  (T, [105.29, 65.96, 30.06, 23.43, 16.29, 14.78, 14.07, 12.31, 11.61, 10.63]),
    "AvgPool": (T, [67.46, 22.08, 10.50, 9.45, 8.79, 8.75, 9.03, 8.65, 8.43, 8.21]),
}
fig, ax = new_fig()
plot_methods(ax, asr, alpha=0.9)
ax.set_xlim(0, 10.5)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax_in = ax.inset_axes([0.33, 0.40, 0.64, 0.56])
plot_methods(ax_in, asr, alpha=0.9, markersize=3)
ax_in.set_xlim(0.9, 5.5)
ax_in.set_ylim(6.5, 25)
ax_in.xaxis.set_major_locator(MultipleLocator(1))
ax_in.yaxis.set_major_locator(MultipleLocator(5))
ax_in.tick_params(labelsize=6, pad=1.5, length=2)
ax_in.grid(True, color="#999999", linestyle="-", alpha=0.4)
ax.indicate_inset_zoom(ax_in, edgecolor="black", alpha=0.3)
finish(fig, ax, "WER (%)", r"%WER ($\downarrow$) with fine-tuned Whisper CTC",
       "ASR_plots_partial_unfreeze", xticks=np.arange(0, 11, 1.0))

# ------------------------------------------- TopicID x SIB-FLEURS x FROZEN WHISPER
fig, ax = new_fig()
plot_methods(ax, {
    "CTC":     ([5.56, 6.25, 7.143, 8.34, 10.0], [71.42, 75.22, 73.70, 76.42, 75.51]),
    "COSD":    ([0.5, 1.0, 2.5, 6.25, 7.143, 8.34, 10.0], [33.86, 64.11, 66.90, 68.65, 69.37, 68.12, 75.51]),
    "DS":      (T, [58.04, 37.14, 68.13, 65.49, 67.83, 68.71, 70.55, 69.63, 72.34, 75.51]),
    "Random":  (T, [24.64, 52.48, 68.30, 72.24, 74.27, 74.25, 71.81, 76.47, 76.88, 74.63]),
    "AvgPool": (T, [62.21, 66.84, 67.91, 66.29, 75.62, 71.31, 71.83, 76.32, 76.16, 77.89]),
})
finish(fig, ax, "UWA (%)", r"%UWA ($\uparrow$) for TD with Whisper CTC (frozen backbone)",
       "TopicID_plots_frozen_backbone")

# ------------------------------------------ SER x IEMOCAP x FINETUNED WHISPER
fig, ax = new_fig()
plot_methods(ax, {
    "CTC":     ([0.638, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0],
                [37.90, 36.26, 41.14, 45.60, 39.28, 44.84, 44.04, 46.90, 46.04, 50.97]),
    "COSD":    (T, [31.06, 37.37, 43.01, 43.88, 41.87, 47.49, 47.51, 45.17, 54.70, 50.97]),
    "DS":      (T, [45.68, 46.76, 43.01, 42.04, 45.51, 48.94, 48.99, 44.80, 48.93, 50.97]),
    "Random":  (T, [33.63, 33.34, 40.74, 48.16, 48.55, 44.45, 44.52, 49.73, 49.43, 49.89]),
    "AvgPool": (T, [53.07, 49.41, 51.49, 49.56, 53.54, 52.05, 49.84, 51.82, 51.84, 53.64]),
})
finish(fig, ax, "UWA (%)", r"%UWA ($\uparrow$) for ER with fine-tuned Whisper CTC",
       "SER_FinetunedWhisper_UWA")

# --------------------------------------------- ST x EUROPARL x FT WHISPER
fig, ax = new_fig()
plot_methods(ax, {
    "CTC":     ([1.4936, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0],
                [21.62, 27.05, 28.31, 29.68, 27.29, 28.85, 29.23, 29.67, 27.72]),
    "COSD":    (T, [1.70, 1.70, 14.59, 22.86, 26.68, 26.58, 28.52, 28.01, 28.51, 27.72]),
    "DS":      (T, [1.58, 1.43, 1.45, 13.17, 21.20, 21.43, 24.71, 25.09, 26.20, 27.72]),
    "Random":  (T, [1.31, 1.35, 1.07, 1.31, 5.86, 10.17, 8.13, 15.05, 13.78, 23.02]),
    "AvgPool": (T, [1.53, 1.60, 13.95, 20.44, 22.27, 25.99, 27.63, 28.37, 28.16, 29.71]),
})
finish(fig, ax, "Normalized BLEU (%)", r"% Normalized BLEU ($\uparrow$) with fine-tuned Whisper CTC",
       "ST_plots_partial_unfreeze")

# ------------------------------------------------ ASR x VoxPopuli x WAV2VEC2
fig, ax = new_fig()
plot_methods(ax, {
    "CTC":     ([3.5, 4.1, 4.6, 5.2, 5.7, 6.1, 6.9, 8.3, 9.4, 10.0],
                [12.84, 12.59, 11.66, 11.14, 10.94, 10.88, 11.46, 10.65, 10.52, 10.54]),
    "COSD":    ([4.0, 4.6, 5.5, 6.4, 7.0, 8.3, 9.5, 10.0],
                [39.78, 22.10, 12.80, 11.02, 10.75, 10.74, 10.24, 10.54]),
    "DS":      ([10.0, 7.2, 5.6, 5.00, 3.3, 2.5, 2.0],
                [10.54, 11.59, 12.81, 12.87, 20.28, 34.36, 65.68]),
    "Random":  ([2.5, 5.0, 6.25, 10.0], [100.84, 60.01, 46.46, 27.51]),
    # NOTE: the original script defined wer_avgpool_test twice; the second
    # definition (below) is the one that was plotted, so it is kept here.
    "AvgPool": ([2.0, 2.5, 3.3, 5.0, 6.25, 10.0], [46.37, 25.87, 17.33, 12.91, 11.45, 10.83]),
})
finish(fig, ax, "WER (%)", r"%WER ($\downarrow$) with Wav2Vec2", "ASR_plots_wav2vec",
       xticks=np.arange(2, 11, 1.0))

print("done")
