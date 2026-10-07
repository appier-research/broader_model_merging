"""One-panel ViT-B-32 bars: coefficient search vs weight optimization.

Reads main_result_vision.csv. No Base / GD-from-base line.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from plot_main_result import FORMATS, PDF_FONT_RC, load

FONT_SIZES = {
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.size": 13,
}

MODEL = "ViT-B-32"
HUE_ORDER = ["Coefficient search", "gradient descent"]
LEGEND_LABELS = {
    "Coefficient search": "Optimize Coefficients",
    "gradient descent": "Optimize Weights",
}
PALETTE = ["#49c6fc", "#0a5cb5"]
FIGSIZE = (4.4, 2.8)
BAR_WIDTH = 0.7


def main() -> None:
    here = Path(__file__).parent
    long, _, _ = load(here / "main_result_vision.csv", only_models=[MODEL])
    long = long[long["Baseline"].isin(HUE_ORDER)].copy()

    sns.set_theme(style="whitegrid", context="paper", rc={**FONT_SIZES, **PDF_FONT_RC})
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    sns.barplot(
        data=long,
        x="Merging Method",
        y="Accuracy",
        hue="Baseline",
        hue_order=HUE_ORDER,
        palette=PALETTE,
        width=BAR_WIDTH,
        errorbar=None,
        ax=ax,
    )

    lo, hi = long["Accuracy"].min(), long["Accuracy"].max()
    ax.set_ylim(max(0, lo - 6), hi + 2.5)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.minorticks_off()
    ax.set_xlabel("")
    ax.set_ylabel("Average accuracy (%)")
    ax.set_title(MODEL)

    if ax.legend_ is not None:
        ax.legend_.remove()
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        [LEGEND_LABELS[s] for s in labels],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    color_of = dict(zip(HUE_ORDER, PALETTE))
    for text, raw in zip(legend.get_texts(), labels):
        text.set_color(color_of[raw])
        text.set_fontweight("bold")

    out = here / "figures" / "vit32_coeff_wo"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in FORMATS:
        path = out.with_suffix(suffix)
        fig.savefig(path, dpi=400)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
