#!/usr/bin/env python3
"""Score vs cost: rows = memory / time / flops, columns = models.

Reads cost units from --results-dir and test_avg_score from --acc-dir (the
main results root). One marker per strategy; the GD strategies' score comes
from the init the profile was taken at (``init`` in the cost unit). ``--rows``
selects which cost rows to draw. Same figure as vision_exp/scripts/plots/plot_cost.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, NullLocator

UNITS_SUBDIR = "units"
PLOTS_SUBDIR = "plots"
METRIC = "test_avg_score"

# Same ColorBrewer blues as vision_exp/scripts/plots/plot_cost.py (coeff / mid / weight).
STRAT_COLOR = {
    "coeff_search": "#49c6fc",
    "bo_search": "#2e7ebc",
    "subspace_gd": "#2e7ebc",
    "weight_gd_lora": "#0a5cb5",
}
# Same panel size and printed font sizes as vision_exp/scripts/plots/plot_cost.py:
# that figure is 3.5in (half-\textwidth) x 3.5in for three model columns, so
# each column is ~1.17in wide; fewer columns give a proportionally narrower
# canvas at the same height.
PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "font.size": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    # tight text-to-axes gaps so the whitespace goes to the panels, not the labels
    "axes.labelpad": 2.0, "axes.titlepad": 3.0, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
}

ARCH_ORDER = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "llama3.2-1b")
STRAT_ORDER = ("coeff_search", "bo_search", "subspace_gd", "weight_gd_lora")
# What the figure shows by default; subspace_gd is profiled but left off the
# plot (--strategies puts it back).
DEFAULT_PLOT_STRATEGIES = ("coeff_search", "bo_search", "weight_gd_lora")
STRAT_LABEL = {
    "coeff_search": "Coeff search",
    "bo_search": "BO search",
    "subspace_gd": "Subspace GD",
    "weight_gd_lora": "Weight GD (LoRA)",
}
STRAT_MARKER = {"coeff_search": "s", "bo_search": "D", "subspace_gd": "^", "weight_gd_lora": "o"}
MARKER_SIZE = 42
FIG_HEIGHT_IN = 3.5
COL_WIDTH_IN = 3.5 / 3  # vision's three-column, half-textwidth canvas
WSPACE, HSPACE = 0.10, 0.02  # constrained-layout gaps, as fractions of a panel

# tick: "plain" | "power10" (log ticks labeled $10^{x}$)
ROW_SPEC = {
    "memory": ("peak_mem_allocated_bytes", 1e-9, "Peak GPU memory (GB)", False, "plain"),
    "time": ("est_wall_s", 1.0 / 60.0, "Time (minutes, log scale)", True, "plain"),
    "flops": ("est_model_flops", 1.0, "FLOPs (log scale)", True, "power10"),
}


def _log_ticks(xs: list, power10: bool) -> list:
    """Two nearby log ticks (decades for FLOPs, 1-2-5 for time). Not forced past the data."""
    lo, hi = min(xs), max(xs)
    if power10:
        e0 = math.floor(math.log10(lo))
        e1 = math.floor(math.log10(hi))
        if e1 <= e0:
            e1 = e0 + 1
        return [10.0 ** e0, 10.0 ** e1]
    ticks = []
    pad_lo, pad_hi = lo * 0.7, hi * 1.4
    for exp in range(math.floor(math.log10(pad_lo)), math.ceil(math.log10(pad_hi)) + 1):
        for m in (1, 2, 5):
            t = m * 10 ** exp
            if pad_lo <= t <= pad_hi:
                ticks.append(t)
    if len(ticks) >= 2:
        return [ticks[0], ticks[-1]]
    return [lo, hi] if lo != hi else [lo]


def _fmt_plain(x, _pos=None) -> str:
    if x >= 10:
        return f"{x:.0f}"
    return f"{x:g}"


def _fmt_power10(x, _pos=None) -> str:
    if x <= 0:
        return ""
    return rf"$10^{{{int(round(math.log10(x)))}}}$"


def _pretty_arch(arch: str) -> str:
    """qwen3-0.6b -> Qwen3-0.6B, llama3.2-1b -> Llama3.2-1B."""
    fam, _, size = arch.partition("-")
    if not size:
        return arch
    fam = fam[:1].upper() + fam[1:]
    return f"{fam}-{size.upper()}"


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
        prev = out.get(key)
        # Prefer the canonical unit name (...__ta__full.json) over extras
        # like ...__full__with_loss.json that share the same key.
        if prev is not None and len(os.path.basename(r["_path"])) >= len(os.path.basename(prev["_path"])):
            continue
        out[key] = r
    return out


def _strategy_score(acc_rec: dict, strategy: str, cost_node: dict):
    strat = acc_rec.get("strategies", {}).get(strategy)
    if strat is None:
        return None
    if strategy == "coeff_search":
        return strat.get(METRIC)
    init = cost_node.get("init")
    node = strat.get(init) if isinstance(strat, dict) and init else None
    return None if node is None else node.get(METRIC)


def _points(cost_recs, acc_idx, method: str, strategies):
    """arch -> strategy -> (cost node, y_percent)."""
    by_arch = {}
    for rec in cost_recs:
        if rec.get("method") != method:
            continue
        arch = rec["arch"]
        key = (tuple(rec["tasks"]), arch, method, str(rec.get("budget", "full")))
        acc_rec = acc_idx.get(key)
        if acc_rec is None:
            print(f"  skip {os.path.basename(rec['_path'])}: no matching main-results unit")
            continue
        slot = by_arch.setdefault(arch, {})
        for strat, node in rec.get("strategies", {}).items():
            if strat not in strategies or not isinstance(node, dict):
                continue
            y = _strategy_score(acc_rec, strat, node)
            if y is None:
                print(f"  skip {arch} {strat} (init={node.get('init')}): no {METRIC} in main results")
                continue
            slot[strat] = (node, 100.0 * y)
    return by_arch


def plot_cost(results_dir: str, acc_dir: str, method: str, rows: list, out: str | None,
              strategies=DEFAULT_PLOT_STRATEGIES):
    strategies = [s for s in STRAT_ORDER if s in strategies]
    if not strategies:
        raise SystemExit(f"--strategies must be a subset of {list(STRAT_ORDER)}")
    cost_recs = _load_units(results_dir)
    acc_idx = _acc_index(_load_units(acc_dir))
    by_arch = _points(cost_recs, acc_idx, method, strategies)
    arches = [a for a in ARCH_ORDER if a in by_arch] + sorted(a for a in by_arch if a not in ARCH_ORDER)
    if not arches:
        raise SystemExit(f"no cost units for method={method} in {results_dir}")
    rows = [r for r in rows if r in ROW_SPEC]
    if not rows:
        raise SystemExit(f"--rows must be a subset of {list(ROW_SPEC)}")

    n_r, n_c = len(rows), len(arches)
    fig_w, fig_h = COL_WIDTH_IN * n_c, FIG_HEIGHT_IN
    # One x label per row, centered: under the middle column when there is one,
    # otherwise on the left-middle axis shifted into the gap (x in axes
    # fraction, so 1 + WSPACE/2 is the gap's center).
    label_ax = n_c // 2 if n_c % 2 else n_c // 2 - 1
    label_kw = {} if n_c % 2 else {"x": 1.0 + WSPACE / 2}
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
                for strat in strategies:
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
                        # linear rows start at 0 so the ratio between strategies
                        # is visible, not just their order
                        x0, x1 = 0.0, hi_x * 1.12
                    ax.set_xlim(x0, x1)
                if log_x:
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_locator(NullLocator())
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    ticks = _log_ticks(xs, power10=(tick == "power10")) if xs else []
                    if ticks:
                        ax.set_xlim(min(x0, ticks[0]), max(x1, ticks[-1]))
                        ax.set_xticks(ticks)
                    fmt = _fmt_power10 if tick == "power10" else _fmt_plain
                    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
                ax.grid(True, which="major", color="0.9", linewidth=0.6)
            if ys:
                lo, hi = min(ys), max(ys)
                pad = max(2.5, 0.14 * (hi - lo))
                axes[i, 0].set_ylim(lo - pad, hi + pad)
            axes[i, label_ax].set_xlabel(xlabel, **label_kw)
        fig.supylabel("Average score (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            # one row when it fits (~1.1in of legend per entry), else two rows
            ncol = len(handles) if fig_w >= 1.1 * len(handles) else (len(handles) + 1) // 2
            fig.legend(
                handles, labels, loc="outside lower center", ncol=ncol,
                frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=0.8,
            )
        fig.get_layout_engine().set(wspace=WSPACE, hspace=HSPACE)
        if out is None:
            out = os.path.join(results_dir, PLOTS_SUBDIR, f"{method}__{'_'.join(rows)}.png")
        _save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/cost_estimation")
    p.add_argument("--acc-dir", default="results", help="Main results root holding test_avg_score")
    p.add_argument("--method", default="ta")
    p.add_argument("--rows", default="memory,time,flops",
                   help="comma list from memory,time,flops")
    p.add_argument("--strategies", default=",".join(DEFAULT_PLOT_STRATEGIES),
                   help=f"comma list from {list(STRAT_ORDER)} (default omits subspace_gd)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = [x.strip() for x in args.rows.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    plot_cost(args.results_dir, args.acc_dir, args.method, rows, args.out, strategies)


if __name__ == "__main__":
    main()
