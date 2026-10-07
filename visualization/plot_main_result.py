"""Visualize merged-model accuracy tables (vision and LLM) as grouped bar charts.

Each CSV is the same shape: a sparse `Merging Method` column crossed with four
`Baseline` variants, scored on one column per model. Layout is one facet per
model, grouped bars of merging method x baseline. The `Base` row is gradient
descent applied directly to the pretrained model -- a reference point, not a
merging baseline -- so it is drawn as a dashed line instead of a bar.

Model columns are read from the header, so adding a model to a CSV needs no
code change. Empty columns, methods, and Base rows are dropped automatically.
A dataset may pin an explicit `models` list to plot a subset of the header.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns

# ordered weakest -> strongest so the bar groups read left to right
BASELINE_ORDER = [
    "Naive baseline",
    "Coefficient search",
    "Subspace upper bound",
    "gradient descent",
]
METHOD_ORDER = ["TA", "TIES", "DARE", "TSV-M"]
# shown in the legend instead of the raw CSV value, so all five entries fit on
# one row at 7in; the keys must stay in sync with BASELINE_ORDER
LEGEND_LABELS = {
    # The shorthand is fine here: Figure 1 spells this out as "Weight
    # optimization" (see plot_figure_1.LABELS), so by the time the reader
    # reaches the results grid the term is established and the short form keeps
    # all five entries on one legend row.
    "gradient descent": "GD",
    # the CSVs still carry the original wording in the Baseline column
    "Subspace upper bound": "Subspace best",
}
REF_LEGEND_LABEL = "GD from pretrained"
# Grey marks the one baseline that never touches the auxiliary data: the naive
# merge fixes its coefficient a priori (merged_avg = 1/N, merged_coeff1 = 1.0),
# and strategies/baselines.py is explicit that its valid numbers are "read-only
# context, not a selection signal". The other three all consume that data, and
# the blue ramp orders them by how much of it the data gets to decide: one
# scalar lambda, then k coefficients, then every weight. Pinned to Figure 1's
# two stops (#49c6fc, #0a5cb5).
#
# Splitting the grey off also buys back the separation a four-step ramp loses:
# three stops span the range at dL* 18/17 instead of four at 13/18/17.
BAR_PALETTE = ["#c2c8ce", "#49c6fc", "#288fd7", "#0a5cb5"]
# Black, and the one mark that sits outside the palette entirely: every hue in
# BAR_PALETTE now carries meaning (grey = no auxiliary data, blue = data, depth
# = how much of the model the data gets to decide), so giving this line any of
# them would enlist it in a code it is not part of. Neutral black reads as
# annotation rather than as a fifth category, and stays legible crossing a bar
# of any depth.
REF_COLOR = "black"
# Points, not scaled by linewidth -- unlike the "--" shorthand, which matplotlib
# stretches by lw to ~7.7pt strokes with 1.9pt gaps at lw 1.2, so dense it reads
# as a near-solid rule. An equal 3pt-on / 3pt-off pitch is unmistakably a dashed
# line at a glance, which is what keeps it from competing now that the colour is
# black. Butt caps: round ones would extend each stroke by lw/2 at both ends and
# eat most of the gap back.
REF_DASH = (0, (3, 3))
REF_WIDTH = 1.2
REF_CAPSTYLE = "butt"
# Over the bars (zorder 1), so the level can be read against every group rather
# than only where no bar reaches it. The dotted form is what keeps it from
# dominating now that the colour no longer does.
REF_ZORDER = 4

ID_COLS = ["Merging Method", "Baseline"]

# raster for quick viewing, vector for the paper
FORMATS = [".png", ".pdf"]

# The figure is included at \textwidth in a two-column paper, so draw the canvas
# at that exact width: scale factor 1.0 means every font size below is the real
# point size on the printed page. Sizing fonts without pinning the width is what
# makes paper figures unreadable -- a 16in canvas squeezed into 7in shrinks 13pt
# type to 5.5pt.
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 2.6
FACET_HEIGHT_IN = 2.35

# true point sizes at scale 1.0, a little under the paper's ~9pt body text
# camera-ready submissions typically reject Type 3 fonts; 42 = TrueType
PDF_FONT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42}

FONT_SIZES = {
    "axes.titlesize": 9,      # facet titles (model names)
    "axes.titleweight": "bold",
    "axes.labelsize": 10,     # row y-labels (accuracy / score)
    "xtick.labelsize": 9.5,   # merging method names
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "font.size": 9.5,
}

# one entry per table; `models` is discovered from the header at load time
# unless the spec pins a list, in which case only those columns are plotted
DATASETS = {
    "vision": {
        "csv": "main_result_vision.csv",
        "ylabel": "Average accuracy (%)",
    },
    "llm": {
        "csv": "main_result_llm.csv",
        "ylabel": "Average score (%)",
        # the CSV also carries Llama-3.2-1B; the main figure shows the Qwen3 scale
        "models": ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"],
    },
}


def load(
    csv_path: Path, only_models: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (long-form rows, Base reference per model, ordered model names).

    `only_models` restricts the plotted columns (in the given order); missing
    names raise so a typo does not silently drop a facet.
    """
    # the files ship tab-separated despite the .csv name; sniff so both work
    sep = "\t" if "\t" in csv_path.read_text().splitlines()[0] else ","
    df = pd.read_csv(csv_path, sep=sep)
    df.columns = [c.strip() for c in df.columns]

    # every column after the two id columns is a model, in file order
    models = [c for c in df.columns if c not in ID_COLS]
    if only_models is not None:
        missing = [m for m in only_models if m not in models]
        if missing:
            raise KeyError(f"{csv_path.name} has no column(s) {missing}")
        models = list(only_models)
    df[ID_COLS] = df[ID_COLS].apply(lambda s: s.str.strip())
    df["Merging Method"] = df["Merging Method"].ffill()
    df = df.dropna(subset=["Baseline"])
    df[models] = df[models].apply(pd.to_numeric, errors="coerce")

    # drop models with no numbers anywhere (a placeholder column not yet filled)
    models = [m for m in models if df[m].notna().any()]

    base_rows = df[df["Merging Method"] == "Base"]
    # a table may ship without a Base row; fall back to all-NaN
    base = (
        base_rows[models].iloc[0]
        if len(base_rows)
        else pd.Series(float("nan"), index=models)
    )
    df = df[df["Merging Method"] != "Base"]

    long = df.melt(
        id_vars=ID_COLS,
        value_vars=models,
        var_name="Model",
        value_name="Accuracy",
    ).dropna(subset=["Accuracy"])

    long["Merging Method"] = long["Merging Method"].replace({"TSVM": "TSV-M"})
    long["Merging Method"] = pd.Categorical(
        long["Merging Method"], METHOD_ORDER, ordered=True
    )
    canon = {b.lower(): b for b in BASELINE_ORDER}
    long["Baseline"] = long["Baseline"].str.lower().map(canon)
    long["Baseline"] = pd.Categorical(long["Baseline"], BASELINE_ORDER, ordered=True)
    long["Model"] = pd.Categorical(long["Model"], models, ordered=True)
    long = long.sort_values(["Model", "Merging Method", "Baseline"])
    return long, base, models


def plot_bars(
    long: pd.DataFrame,
    base: pd.Series,
    models: list[str],
    out_stem: Path,
    ylabel: str = "Average accuracy (%)",
) -> None:
    # "paper" context, then override with explicit point sizes so the result is
    # independent of seaborn's context scaling
    sns.set_theme(style="whitegrid", context="paper", rc={**FONT_SIZES, **PDF_FONT_RC})
    present_methods = [m for m in METHOD_ORDER if m in set(long["Merging Method"])]
    present_baselines = [b for b in BASELINE_ORDER if b in set(long["Baseline"])]

    grid = sns.catplot(
        data=long,
        kind="bar",
        x="Merging Method",
        y="Accuracy",
        hue="Baseline",
        col="Model",
        order=present_methods,
        hue_order=present_baselines,
        col_order=models,
        palette=BAR_PALETTE[: len(present_baselines)],
        height=FACET_HEIGHT_IN,
        # catplot sizes per facet; divide the target width across the columns so
        # the assembled figure lands on FIG_WIDTH_IN
        aspect=(FIG_WIDTH_IN / max(1, len(models))) / FACET_HEIGHT_IN,
        legend_out=False,
        errorbar=None,
    )

    # pad the axis around the actual data plus any reference line
    values = pd.concat([long["Accuracy"], base.dropna()])
    lo, hi = values.min(), values.max()
    grid.set(ylim=(max(0, lo - 6), hi + 2.5))
    grid.set_axis_labels("", ylabel)
    grid.set_titles("{col_name}")

    # catplot hoists the hue legend off the axes, so read the handles from it
    bar_handles = list(grid.legend.legend_handles)
    bar_labels = [
        LEGEND_LABELS.get(t.get_text(), t.get_text())
        for t in grid.legend.get_texts()
    ]

    has_ref = False
    for model, ax in grid.axes_dict.items():
        ref = base.get(model, float("nan"))
        if pd.notna(ref):
            has_ref = True
            ax.axhline(
                ref,
                color=REF_COLOR,
                linestyle=REF_DASH,
                linewidth=REF_WIDTH,
                dash_capstyle=REF_CAPSTYLE,
                zorder=REF_ZORDER,
            )

    # rebuild the legend as a single figure-level row under the facets
    if grid.legend is not None:
        grid.legend.remove()
    handles, labels = bar_handles, list(bar_labels)
    if has_ref:
        handles = handles + [
            Line2D(
                [], [], color=REF_COLOR, linestyle=REF_DASH,
                linewidth=REF_WIDTH, dash_capstyle=REF_CAPSTYLE,
            )
        ]
        labels = labels + [REF_LEGEND_LABEL]
    grid.figure.legend(
        handles,
        labels,
        # "outside lower center" is understood by constrained_layout, which then
        # reserves a strip for the legend inside the fixed canvas instead of
        # letting it fall off the bottom edge
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # bbox_inches="tight" grows the canvas by whatever sits outside the axes
    # (y label, legend row), so measure that overhead once and shrink the figure
    # to compensate -- the saved file then really is FIG_WIDTH_IN wide and the
    # font sizes above survive \includegraphics at scale 1.0
    # Save the exact canvas size rather than letting bbox_inches="tight" pick it:
    # a tight bbox includes the legend and y label, so the file would come out
    # wider than FIG_WIDTH_IN and \includegraphics would scale the fonts back
    # down. constrained_layout packs the decorations inside the fixed canvas.
    fig = grid.figure
    fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN, forward=True)
    fig.set_layout_engine("constrained")

    for suffix in FORMATS:
        path = out_stem.with_suffix(suffix)
        # dpi is ignored by the vector backend but harmless to pass
        fig.savefig(path, dpi=400)
        print(f"wrote {path}")
    plt.close(fig)


def plot_stacked(
    rows: list[tuple[pd.DataFrame, pd.Series, list[str], str]],
    out_stem: Path,
) -> None:
    """Same catplot as plot_bars, vision row then LLM row, one legend."""
    sns.set_theme(style="whitegrid", context="paper", rc={**FONT_SIZES, **PDF_FONT_RC})
    frames, titles, ylims, refs = [], {}, {}, {}
    n_col = max(len(models) for _, _, models, _ in rows)
    present_methods, present_baselines = [], []
    for r, (long, base, models, _) in enumerate(rows):
        part = long.copy()
        part["_row"] = r
        part["_col"] = [f"c{models.index(m)}" for m in part["Model"]]
        frames.append(part)
        values = pd.concat([long["Accuracy"], base.dropna()])
        lo, hi = values.min(), values.max()
        ylims[r] = (max(0, lo - 6), hi + 2.5)
        for c, model in enumerate(models):
            titles[(r, f"c{c}")] = model
            refs[(r, f"c{c}")] = base.get(model, float("nan"))
        for m in METHOD_ORDER:
            if m in set(long["Merging Method"]) and m not in present_methods:
                present_methods.append(m)
        for b in BASELINE_ORDER:
            if b in set(long["Baseline"]) and b not in present_baselines:
                present_baselines.append(b)

    grid = sns.catplot(
        data=pd.concat(frames, ignore_index=True),
        kind="bar",
        x="Merging Method",
        y="Accuracy",
        hue="Baseline",
        row="_row",
        col="_col",
        order=present_methods,
        hue_order=present_baselines,
        row_order=list(range(len(rows))),
        col_order=[f"c{i}" for i in range(n_col)],
        palette=BAR_PALETTE[: len(present_baselines)],
        height=FACET_HEIGHT_IN,
        aspect=(FIG_WIDTH_IN / max(1, n_col)) / FACET_HEIGHT_IN,
        legend_out=False,
        errorbar=None,
        sharey=False,
    )
    grid.set_axis_labels("", "")
    grid.set_titles(col_template="", row_template="")
    for r, (*_, ylabel) in enumerate(rows):
        grid.axes[r, 0].set_ylabel(ylabel)

    bar_handles = list(grid.legend.legend_handles)
    bar_labels = [
        LEGEND_LABELS.get(t.get_text(), t.get_text())
        for t in grid.legend.get_texts()
    ]
    has_ref = False
    for key, ax in grid.axes_dict.items():
        if key not in titles:
            ax.set_visible(False)
            continue
        ax.set_title(titles[key])
        ax.set_ylim(*ylims[key[0]])
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.minorticks_off()
        ax.tick_params(axis="x", labelbottom=True)
        if key[1] != "c0":
            ax.tick_params(axis="y", labelleft=False)
        ref = refs[key]
        if pd.notna(ref):
            has_ref = True
            ax.axhline(
                ref,
                color=REF_COLOR,
                linestyle=REF_DASH,
                linewidth=REF_WIDTH,
                dash_capstyle=REF_CAPSTYLE,
                zorder=REF_ZORDER,
            )

    if grid.legend is not None:
        grid.legend.remove()
    handles, labels = bar_handles, list(bar_labels)
    if has_ref:
        handles = handles + [
            Line2D(
                [], [], color=REF_COLOR, linestyle=REF_DASH,
                linewidth=REF_WIDTH, dash_capstyle=REF_CAPSTYLE,
            )
        ]
        labels = labels + [REF_LEGEND_LABEL]
    grid.figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig = grid.figure
    fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN * 2, forward=True)
    fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(hspace=0.12, wspace=0.08)
    for suffix in FORMATS:
        path = out_stem.with_suffix(suffix)
        fig.savefig(path, dpi=400)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="which table to plot (default: all = vision over LLM)",
    )
    parser.add_argument("--outdir", type=Path, default=here / "figures")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    loaded = []

    for name in names:
        spec = DATASETS[name]
        csv_path = here / spec["csv"]
        if not csv_path.exists():
            print(f"skipping {name}: {csv_path.name} not found")
            continue

        long, base, models = load(csv_path, spec.get("models"))
        dropped = sorted(set(METHOD_ORDER) - set(long["Merging Method"].dropna()))
        if dropped:
            print(f"[{name}] no data for: {', '.join(dropped)}")
        if base.dropna().empty:
            print(f"[{name}] Base row empty -- omitting the reference line")

        ylabel = spec.get("ylabel", "Average accuracy (%)")
        if args.dataset == "all":
            loaded.append((long, base, models, ylabel))
        else:
            plot_bars(
                long,
                base,
                models,
                args.outdir / f"main_result_{name}_bars",
                ylabel,
            )

    if loaded:
        plot_stacked(loaded, args.outdir / "main_result_bars")


if __name__ == "__main__":
    main()
