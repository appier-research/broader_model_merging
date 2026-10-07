#!/usr/bin/env python3
"""Accuracy vs cost: rows = memory / time / flops, columns = models.

Reads cost units from --results-dir and test_avg_acc from --acc-dir (main_exp).
One marker per strategy. ``--rows`` selects which cost rows to draw.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cost import SUBSPACE_GD_INIT, weight_gd_init
from src.experiment import UNITS_SUBDIR, PLOTS_SUBDIR

# Same ColorBrewer blues / paper rc as scripts/plots/plot.py.
STRAT_COLOR = {
    "coeff_search": "#6aaed6",
    "subspace_gd": "#2e7ebc",
    "weight_gd": "#084a91",
}
# Canvas is half-\textwidth (~3.5in). Fonts below are printed point sizes.
PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "font.size": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
}

ARCH_ORDER = ("vit-b-32", "vit-b-16", "vit-l-14")
STRAT_ORDER = ("coeff_search", "subspace_gd", "weight_gd")
STRAT_LABEL = {
    "coeff_search": "Coeff search",
    "subspace_gd": "Subspace GD",
    "weight_gd": "Weight GD",
}
STRAT_MARKER = {"coeff_search": "s", "subspace_gd": "^", "weight_gd": "o"}
MARKER_SIZE = 42
FIG_WIDTH_IN = 3.5

# tick: "plain" | "power10" (log ticks labeled $10^{x}$)
ROW_SPEC = {
    "memory": ("peak_mem_allocated_bytes", 1e-9, "Peak GPU memory (GB)", False, "plain"),
    "time": ("est_wall_s", 1.0 / 60.0, "Time (min)", True, "plain"),
    "flops": ("est_model_flops", 1.0, "FLOPs", True, "power10"),
}


def _nice_log_ticks(xs: list[float], power10: bool, max_n: int = 2) -> list[float]:
    pos = [x for x in xs if x > 0]
    if not pos:
        return []
    lo, hi = min(pos) * 0.7, max(pos) * 1.4
    e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    if power10:
        ticks = [10.0 ** e for e in range(e0, e1 + 1)]
    else:
        ticks = []
        for exp in range(e0, e1 + 1):
            for m in (1, 2, 5):
                t = m * 10 ** exp
                if lo <= t <= hi:
                    ticks.append(t)
        if not ticks:
            ticks = [min(pos), max(pos)]
    if len(ticks) > max_n:
        idx = [round(i * (len(ticks) - 1) / (max_n - 1)) for i in range(max_n)]
        ticks = [ticks[k] for k in idx]
    return ticks


def _fmt_plain(x, _pos=None) -> str:
    if x >= 10:
        return f"{x:.0f}"
    return f"{x:g}"


def _fmt_power10(x, _pos=None) -> str:
    if x <= 0:
        return ""
    return rf"$10^{{{int(round(math.log10(x)))}}}$"


def _pretty_arch(arch: str) -> str:
    parts = arch.replace("_", "-").split("-")
    if parts and parts[0].lower() == "vit":
        return "ViT-" + "-".join(p.upper() for p in parts[1:])
    return arch


def _save(fig, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=400)
    pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved plot -> {out}")
    print(f"Saved plot -> {pdf}")


def _load_units(root: str) -> list:
    paths = sorted(glob.glob(os.path.join(root, UNITS_SUBDIR, "*.json")))
    recs = []
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        rec["_path"] = p
        recs.append(rec)
    return recs


def _acc_index(recs: list) -> dict:
    out = {}
    for r in recs:
        key = (tuple(r["tasks"]), r["arch"], r["method"], str(r.get("budget", "full")))
        out[key] = r
    return out


def _strategy_acc(acc_rec: dict, strategy: str, method: str):
    strat = acc_rec.get("strategies", {}).get(strategy)
    if strat is None:
        return None
    if strategy == "coeff_search":
        return strat.get("test_avg_acc")
    init = SUBSPACE_GD_INIT if strategy == "subspace_gd" else weight_gd_init(method)
    node = strat.get(init) if isinstance(strat, dict) else None
    return None if node is None else node.get("test_avg_acc")


def _points(cost_recs, acc_idx, method: str):
    """arch -> strategy -> (x_raw dict, y_percent)."""
    by_arch = {}
    for rec in cost_recs:
        if rec.get("method") != method:
            continue
        arch = rec["arch"]
        key = (tuple(rec["tasks"]), arch, method, str(rec.get("budget", "full")))
        acc_rec = acc_idx.get(key)
        if acc_rec is None:
            print(f"  skip {os.path.basename(rec['_path'])}: no matching main_exp unit")
            continue
        slot = by_arch.setdefault(arch, {})
        for strat, node in rec.get("strategies", {}).items():
            if strat not in STRAT_ORDER or not isinstance(node, dict):
                continue
            y = _strategy_acc(acc_rec, strat, method)
            if y is None:
                print(f"  skip {arch} {strat}: no test_avg_acc in main_exp")
                continue
            slot[strat] = (node, 100.0 * y)
    return by_arch


def plot_cost(results_dir: str, acc_dir: str, method: str, rows: list, out: str | None):
    cost_recs = _load_units(results_dir)
    acc_idx = _acc_index(_load_units(acc_dir))
    by_arch = _points(cost_recs, acc_idx, method)
    arches = [a for a in ARCH_ORDER if a in by_arch] or sorted(by_arch)
    if not arches:
        raise SystemExit(f"no cost units for method={method} in {results_dir}")
    rows = [r for r in rows if r in ROW_SPEC]
    if not rows:
        raise SystemExit(f"--rows must be a subset of {list(ROW_SPEC)}")

    n_r, n_c = len(rows), len(arches)
    fig_w = fig_h = FIG_WIDTH_IN
    mid = n_c // 2
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            n_r, n_c, figsize=(fig_w, fig_h), squeeze=False, sharey="row",
            layout="constrained",
        )
        for j, arch in enumerate(arches):
            axes[0, j].set_title(_pretty_arch(arch))
        for i, row in enumerate(rows):
            key, scale, xlabel, log_x, tick = ROW_SPEC[row]
            ys = []
            for j, arch in enumerate(arches):
                ax = axes[i, j]
                xs = []
                for strat in STRAT_ORDER:
                    pair = by_arch[arch].get(strat)
                    if pair is None:
                        continue
                    node, y = pair
                    if node.get(key) is None:
                        continue
                    x = float(node[key]) * scale
                    xs.append(x)
                    ys.append(y)
                    ax.scatter(
                        x, y, s=MARKER_SIZE, zorder=3,
                        color=STRAT_COLOR[strat], marker=STRAT_MARKER[strat],
                        label=STRAT_LABEL[strat] if j == 0 and i == 0 else None,
                    )
                if xs:
                    lo_x, hi_x = min(xs), max(xs)
                    if log_x:
                        x0, x1 = lo_x * 0.7, hi_x * 1.45
                    else:
                        span = (hi_x - lo_x) or max(hi_x, 1.0)
                        x0, x1 = lo_x - 0.22 * span, hi_x + 0.12 * span
                    ax.set_xlim(x0, x1)
                if log_x:
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_locator(NullLocator())
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    ticks = [t for t in _nice_log_ticks(xs, power10=(tick == "power10"))
                             if xs and x0 <= t <= x1]
                    if not ticks and xs:
                        ticks = [min(xs), max(xs)] if min(xs) != max(xs) else [min(xs)]
                    if ticks:
                        ax.set_xticks(ticks)
                    fmt = _fmt_power10 if tick == "power10" else _fmt_plain
                    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                ax.grid(True, which="major", color="0.9", linewidth=0.6)
            if ys:
                lo, hi = min(ys), max(ys)
                pad = max(2.5, 0.14 * (hi - lo))
                axes[i, 0].set_ylim(lo - pad, hi + pad)
            axes[i, mid].set_xlabel(xlabel)
        fig.supylabel("Average accuracy (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="outside lower center", ncol=len(handles),
                frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
            )
        fig.get_layout_engine().set(wspace=0.12, hspace=0.08)
        if out is None:
            out = os.path.join(results_dir, PLOTS_SUBDIR, f"{method}__{'_'.join(rows)}.png")
        _save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/cost_estimation")
    p.add_argument("--acc-dir", default="results/main_exp")
    p.add_argument("--method", default="ta")
    p.add_argument("--rows", default="memory,time,flops",
                   help="comma list from memory,time,flops")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = [x.strip() for x in args.rows.split(",") if x.strip()]
    plot_cost(args.results_dir, args.acc_dir, args.method, rows, args.out)


if __name__ == "__main__":
    main()
