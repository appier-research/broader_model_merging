#!/usr/bin/env python3
"""Distribution of TEST scores over directional_sampling draws, per unit.

Reads the unit JSONs a DS_EVAL_TEST=1 run writes (every draw carries
``test_avg_score``; see src/strategies/directional_sampling.py) and draws one
panel per (arch, method) cell:

  * histogram of ``samples[*].test_avg_score`` (shown x100, like every other
    score axis in this repo's figures);
  * a vertical line at the init point's test score (drawn as theta*_sub: the best
    point found inside the merge subspace) -- the reference every
    draw is read against. By default that is the score the init's source cell
    recorded (``init_source_test_avg_score``: bo_search / coeff_search, the
    same number the main results report); ``--center reeval`` uses the run's
    own re-evaluation (``center_test_avg_score``) instead. The printed
    win rate is recomputed against whichever line is drawn.

Panels are small multiples laid out like the paper's other figures: models
across the top (column titles), merging methods down the side (row labels).
The side of each panel to the right of the init point is faintly shaded and
the panel's legend gives the win rate (share of draws whose test score beats
the init point); a second dashed line marks what validation would pick -- the
mean test acc of the valid top-5 by default, the single valid-best draw with
``--selected top1``. Panels of the same model share one x-range, and every
column spans the same width, so bins look identical across panels.
``--methods ta,dare`` gives the main-paper figure; the default (every method
found) the appendix one.

Style follows scripts/plots/plot_init_scatter.py / plot_cost.py (same PAPER_RC,
the main figures' ColorBrewer blue, percent axes, 400-dpi PNG + PDF);
fixed 0.5-pt bins drawn at 90% width.

Examples
--------
    python scripts/plots/plot_ds_test_dist.py
    python scripts/plots/plot_ds_test_dist.py --selected top1 --methods ta,dare               # main paper
    python scripts/plots/plot_ds_test_dist.py --selected top1 --methods ta,dare --layout row  # single row
    python scripts/plots/plot_ds_test_dist.py --selected top1                                 # appendix
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

UNITS_SUBDIR = "units"
PLOTS_SUBDIR = "plots"

# One colour for every panel: the method is already the panel title, so hue
# carries no information here. ColorBrewer Blues at 0.7 -- the middle stop of
# visualization/plot_main_result.py's BAR_PALETTE (and plot_cost.py's
# subspace_gd blue), so this sits in the same ramp as the main figures.
C_HIST = "#6aaed6"   # Blues 0.5: the bars
C_TOP5 = "#0a5cb5"   # the main figure's blue: the valid-selected draw (bright enough to read against black)
METHOD_ORDER = ("ta", "ties", "dare", "tsvm")
METHOD_LABEL = {"ta": "TA", "ties": "TIES", "dare": "DARE", "tsvm": "TSV-M"}
ARCH_ORDER = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "llama3.2-1b")
ARCH_TITLE = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B",
              "llama3.2-1b": "Llama-3.2-1B"}

C_INIT = "black"     # init-point reference line
# The init point is the best point found *inside the merge subspace* (BO best /
# lambda*); the draws leave that subspace along the gradient axis.
# Same paper rc as plot_init_scatter.py: the canvas is the printed width, so
# fonts equal their on-page point size.
PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "font.size": 9, "axes.titlepad": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
}
PANEL_W, PANEL_H = 2.6, 2.0   # per panel; two panels side by side ~ one \linewidth
ROW_PANEL_W = 1.75   # --layout row: four panels ~ 7 in, scaled to \textwidth in LaTeX

UNIT_RE = re.compile(r"^(?P<n>\d+)_tasks_(?P<tasks>.+?)__(?P<arch>[^_]+(?:_[^_]+)*?)__(?P<method>[a-z]+)__(?P<budget>[^_]+)\.json$")


def _load_cells(results_dir: str, arch: Optional[str], method: Optional[str], center_mode: str = "recorded",
                relative: bool = False) -> List[dict]:
    """One record per (unit, init) directional_sampling cell that has per-draw
    test scores."""
    cells = []
    for path in sorted(glob.glob(os.path.join(results_dir, UNITS_SUBDIR, "*.json"))):
        m = UNIT_RE.match(os.path.basename(path))
        if not m:
            continue
        if arch and m["arch"] != arch:
            continue
        if method and m["method"] != method:
            continue
        with open(path) as f:
            rec = json.load(f)
        ds = rec.get("strategies", {}).get("directional_sampling", {})
        for init, cell in ds.items():
            scores = [s.get("test_avg_score") for s in cell.get("samples", [])]
            scores = [100.0 * x for x in scores if x is not None]
            if not scores:
                continue  # an old-protocol cell: only the top-5 have test scores
            reeval = cell.get("center_test_avg_score")
            recorded = cell.get("init_source_test_avg_score")
            center = (recorded if recorded is not None else reeval) if center_mode == "recorded" else \
                     (reeval if reeval is not None else recorded)
            top5 = [t["test_avg_score"] for t in cell.get("top5", []) if t.get("test_avg_score") is not None]
            if relative and center is not None:
                scores = [x - 100.0 * center for x in scores]
                top5 = [t - center for t in top5]
                center = 0.0
            cells.append({
                "arch": m["arch"], "method": m["method"], "init": init, "path": path,
                "scores": scores,
                "center": None if center is None else 100.0 * center,
                "center_kind": ("recorded" if (recorded is not None and center_mode == "recorded") or reeval is None
                                else "re-evaluated") if center is not None else None,
                "top5": [100.0 * t for t in top5],
                "dist": cell.get("test_dist") or {},
            })
    return cells


def _xlim(cells: List[dict], span: Optional[float] = None):
    """x-range covering the cells' draws and init points. With ``span`` the
    window is widened (symmetrically) to that width, so every panel of a
    figure gets the same scale and identical-looking bins."""
    lo = min(min(c["scores"]) for c in cells)
    hi = max(max(c["scores"]) for c in cells)
    for c in cells:
        if c["center"] is not None:
            lo, hi = min(lo, c["center"]), max(hi, c["center"])
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    if span is not None and span > hi - lo:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - span / 2, mid + span / 2
    return lo, hi


def _bin_edges(scores: List[float]):
    """0.5-pt bins aligned to multiples of 0.5, so every panel in the figure
    uses the same resolution."""
    start = math.floor(min(scores) / 0.5) * 0.5
    stop = math.ceil(max(scores) / 0.5) * 0.5
    return np.arange(start, stop + 0.25, 0.5)


def _panel(ax, c: dict, xlim, selected: str = "top5mean", headroom: float = 1.25,
           legend_fontsize: float = 8.5) -> Optional[float]:
    """Draw one panel; returns the win rate (None without an init point)."""
    edges = _bin_edges(c["scores"])
    ax.hist(c["scores"], bins=edges, color=C_HIST, rwidth=0.9, edgecolor="none", zorder=3)
    for sp in ax.spines.values():
        sp.set_zorder(6)
    ymax = ax.get_ylim()[1]

    if c["center"] is not None:
        # No label on the line itself: the figure legend names it, and a label
        # here would collide with the panel legend whenever the init point
        # sits near the right edge.
        ax.axvline(c["center"], color=C_INIT, linewidth=1.2, linestyle="-", zorder=4)
        # Win rate = share of draws to the right of the init point. That side of
        # the panel is faintly shaded and the panel's own legend names the share.
        win = sum(x > c["center"] for x in c["scores"]) / len(c["scores"])
        ax.axvspan(c["center"], xlim[1], facecolor=C_HIST, alpha=0.12, edgecolor="none", zorder=1)
        # Short label ("95%") in whichever top corner has less histogram
        # mass under it; the figure legend explains the shading.
        mid = 0.5 * (xlim[0] + xlim[1])
        counts, edges = np.histogram(c["scores"], bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        left_peak = max([n for n, x in zip(counts, centers) if x < mid], default=0)
        right_peak = max([n for n, x in zip(counts, centers) if x >= mid], default=0)
        loc = "upper right" if left_peak > right_peak else "upper left"
        label = f"win rate {100 * win:.0f}%"
        leg = ax.legend(handles=[Patch(facecolor=C_HIST, alpha=0.25, edgecolor="none", label=label)],
                        loc=loc, fontsize=legend_fontsize, handlelength=1.2,
                        handletextpad=0.5, borderaxespad=0.3, frameon=True, fancybox=True, framealpha=1.0,
                        facecolor="white", edgecolor="0.8")
        leg.get_frame().set_linewidth(0.6)
    else:
        win = None

    # The draw that validation would actually pick (rank 1 by valid score),
    # as a second reference line: how far the practical selection lands from
    # the init point and from the bulk of the distribution.
    if c["top5"]:
        x_sel = c["top5"][0] if selected == "top1" else sum(c["top5"]) / len(c["top5"])
        ax.axvline(x_sel, color=C_TOP5, linewidth=1.2, linestyle="--", zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax * headroom)   # headroom so the panel legend sits above the bars
    ax.grid(True, axis="y", color="0.9", linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(length=2.5, width=0.8, colors="0.25")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.25")
    return win


def plot(cells: List[dict], out: str, methods_order: Optional[List[str]], title: Optional[str], sharex: bool,
         relative: bool = False, selected: str = "top5mean", layout: str = "grid") -> None:
    archs = sorted({c["arch"] for c in cells},
                   key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else 99)
    methods = [m for m in (methods_order or METHOD_ORDER) if any(c["method"] == m for c in cells)]
    methods += sorted({c["method"] for c in cells} - set(methods))
    # Same convention as the paper's other figures: models across the top,
    # methods down the side.
    by_key = {(c["arch"], c["method"]): c for c in cells}
    if layout == "row":
        return _plot_row(cells, archs, methods, by_key, out, title, relative, selected)
    nrow, ncol = len(methods), len(archs)
    common_xlim = _xlim(cells) if sharex else None
    # One x-range per column (= per model), widened to the widest column's
    # span: panels of the same model share an axis, and a 0.25-pt bin is the
    # same width on paper in every panel.
    col_cells = {arch: [c for c in cells if c["arch"] == arch] for arch in archs}
    span = max(b - a for a, b in (_xlim(cc) for cc in col_cells.values()))
    col_xlim = {arch: _xlim(cc, span) for arch, cc in col_cells.items()}

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(PANEL_W * ncol, PANEL_H * nrow), squeeze=False,
                                 sharex=sharex)
        for i, method in enumerate(methods):
            for j, arch in enumerate(archs):
                ax = axes[i][j]
                c = by_key.get((arch, method))
                if c is None:
                    ax.axis("off")
                    continue
                _panel(ax, c, common_xlim or col_xlim[arch], selected)
                if i == 0:
                    ax.set_title(ARCH_TITLE.get(arch, arch), pad=9)
                if j == 0:
                    ax.set_ylabel("Count")
                    if nrow > 1 or True:
                        ax.annotate(METHOD_LABEL.get(method, method), xy=(0, 0.5), xycoords="axes fraction",
                                    xytext=(-46, 0), textcoords="offset points", rotation=90,
                                    ha="center", va="center", fontsize=10, fontweight="bold")
                if i == nrow - 1 or not sharex:
                    ax.set_xlabel("Test score − subspace best (pt)" if relative else "Test score (%)")

        handles = [
            Line2D([0], [0], color=C_INIT, linewidth=1.2, linestyle="-", label="Subspace best"),
            Line2D([0], [0], color=C_TOP5, linewidth=1.2, linestyle="--",
                   label="Selected by auxiliary data acc" + ("" if selected == "top1" else " (top-5 mean)")),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handletextpad=0.5, columnspacing=1.6, handlelength=1.6, fontsize=9)
        if title:
            fig.suptitle(title, fontweight="bold")
        fig.tight_layout(rect=(0.03, 0.04, 1, 1), h_pad=1.6)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=400, bbox_inches="tight")
        fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"saved {out} (+ .pdf)")


def _plot_row(cells, archs, methods, by_key, out, title, relative, selected) -> None:
    """Single-row variant for a one-column paper: panels ordered model-major
    (all methods of the first model, then the next model), the method as each
    panel's title and the model written once above its group of panels. The
    x-range is shared within a model, as in the grid layout."""
    order = [(a, m) for a in archs for m in methods if (a, m) in by_key]
    n = len(order)
    col_cells = {a: [c for c in cells if c["arch"] == a] for a in archs}
    span = max(b - lo for lo, b in (_xlim(cc) for cc in col_cells.values()))
    col_xlim = {a: _xlim(cc, span) for a, cc in col_cells.items()}

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, n, figsize=(ROW_PANEL_W * n, 2.1), squeeze=False)
        for j, (arch, method) in enumerate(order):
            ax = axes[0][j]
            # Narrow panels: more headroom so the win-rate legend clears the bars.
            _panel(ax, by_key[(arch, method)], col_xlim[arch], selected, headroom=1.35, legend_fontsize=7.5)
            ax.set_title(METHOD_LABEL.get(method, method), fontweight="normal", pad=4, fontsize=9.5)
            ax.set_xlabel("Test score − subspace best (pt)" if relative else "Test score (%)")
            if j == 0:
                ax.set_ylabel("Count")
        fig.tight_layout(rect=(0, 0.1, 1, 0.9), w_pad=1.2)
        # Model labels: one per group of panels, centred over the group and
        # sitting just above the method titles.
        for arch in archs:
            idx = [j for j, (a, _) in enumerate(order) if a == arch]
            if not idx:
                continue
            p0, p1 = axes[0][idx[0]].get_position(), axes[0][idx[-1]].get_position()
            fig.text(0.5 * (p0.x0 + p1.x1), p0.y1 + 0.105, ARCH_TITLE.get(arch, arch), ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
        handles = [
            Line2D([0], [0], color=C_INIT, linewidth=1.2, linestyle="-", label="Subspace best"),
            Line2D([0], [0], color=C_TOP5, linewidth=1.2, linestyle="--",
                   label="Selected by auxiliary data acc" + ("" if selected == "top1" else " (top-5 mean)")),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0),
                   handletextpad=0.5, columnspacing=1.6, handlelength=1.6, fontsize=9)
        if title:
            fig.suptitle(title, fontweight="bold")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=400, bbox_inches="tight")
        fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"saved {out} (+ .pdf)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/directional_sampling_test_alpha0.05",
                   help="Results root holding units/*.json from DS_EVAL_TEST=1 runs")
    p.add_argument("--archs", default=None, help="Comma-separated arch tags (default: every arch found)")
    p.add_argument("--methods", default=None,
                   help="Comma-separated methods, in row order (default: every method found, ta,ties,dare,tsvm). "
                        "E.g. --methods ta,dare for the main-paper figure.")
    p.add_argument("--center", choices=("recorded", "reeval"), default="recorded",
                   help="Which init-point test score to mark: 'recorded' = the init's source cell (bo_search / "
                        "coeff_search; default), 'reeval' = the run's own re-evaluation")
    p.add_argument("--relative", action="store_true",
                   help="Plot test score minus the init point's (in points) instead of the absolute score: the "
                        "init line sits at 0 in every panel and one shared x-axis fits all of them")
    p.add_argument("--selected", choices=("top5mean", "top1"), default="top5mean",
                   help="Which valid-selected point the solid/dashed blue line marks: the mean test acc of the "
                        "valid top-5 (what the tables report; default) or the single valid-best draw")
    p.add_argument("--layout", choices=("grid", "row"), default="grid",
                   help="grid: models as columns, methods as rows (default); row: one row of panels, "
                        "model-major, for a single-column layout")
    p.add_argument("--sharex", action="store_true",
                   help="Identical x-limits for every panel (default: each panel is centred on its own draws "
                        "but all share the widest panel's span)")
    p.add_argument("--out", default=None,
                   help="Output PNG (default: <results-dir>/plots/test_dist[__<methods>].png)")
    p.add_argument("--title", default=None, help="Optional figure-level title (default: none)")
    args = p.parse_args()

    archs = args.archs.split(",") if args.archs else None
    methods = args.methods.split(",") if args.methods else None
    sharex = args.sharex or args.relative
    cells = [c for c in _load_cells(args.results_dir, None, None, args.center, args.relative)
             if (archs is None or c["arch"] in archs) and (methods is None or c["method"] in methods)]
    if not cells:
        raise SystemExit(f"no directional_sampling cells with per-draw test scores under {args.results_dir}/units")
    for c in cells:
        d, t5 = c["dist"], c["top5"]
        beat = (sum(x > c["center"] for x in c["scores"]) / len(c["scores"])) if c["center"] is not None else float("nan")
        print(f"{c['arch']:12s} {c['method']:5s} init={c['init']:11s} n={len(c['scores']):3d}  "
              f"W*_test={c['center'] if c['center'] is not None else float('nan'):.2f}% ({c['center_kind']})  "
              f"median={100 * d.get('median', float('nan')):.2f}%  "
              f"win_rate={beat:.3f}  "
              f"top5_by_valid mean={sum(t5) / len(t5) if t5 else float('nan'):.2f}% "
              f"best={max(t5) if t5 else float('nan'):.2f}%")

    out = args.out
    if out is None:
        tag = ("test_dist" + (f"__{'-'.join(methods)}" if methods else "") + (f"__{'-'.join(archs)}" if archs else "")
               + ("__relative" if args.relative else "") + ("__top1" if args.selected == "top1" else "")
               + ("__row" if args.layout == "row" else ""))
        out = os.path.join(args.results_dir, PLOTS_SUBDIR, tag + ".png")
    plot(cells, out, methods, args.title, sharex, args.relative, args.selected, args.layout)


if __name__ == "__main__":
    main()
