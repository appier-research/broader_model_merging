#!/usr/bin/env python3
"""Plot the merge-experiment results produced by merge_eval.py.

Results live as one JSON per atomic unit under <results-dir>/units/:

    ...__<budget>.json              # full (no seed suffix)
    ...__<budget>__seed{N}.json     # non-full data_scaling draws

data_scaling plots mean ± std (shaded) across seeds at each budget.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

DEFAULT_METRIC = "test_avg_acc"
DEFAULT_RESULTS_DIR = "results/main_exp"

# Subfolders inside the results root (mirrors src/experiment.py):
#   units/  per-unit JSONs we read;  plots/  figures we write.
UNITS_SUBDIR = "units"
PLOTS_SUBDIR = "plots"

# Shared with visualization/plot_main_result.py: ColorBrewer Blues at 0.5 / 0.7 /
# 0.9 (the three strategy slots on the main-result bars) and the same paper rc.
STRAT_COLOR = {
    "coeff_search": "#49c6fc",
    "subspace_gd": "#288fd7",
    "weight_gd": "#0a5cb5",
    "weight_gd_l2_sp": "#750014",
}
PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
}
SCALE_FIGSIZE = (7.5, 2.6)  # two-arch pair
# one panel = half the pair, same height, so a 2-task figure matches one 9-task subfigure
SCALE_FIGSIZE_ONE = (SCALE_FIGSIZE[0] / 2, SCALE_FIGSIZE[1])

# One consistent colour per init point, one linestyle/hatch per strategy family.
# Used by the full-comparison bars; data_scaling colors by strategy instead.
INIT_COLOR = {
    "pretrained": "mediumpurple",
    "avg": "teal",
    "merged": "darkorange",
    "coeff_best": "steelblue",
}
COEFF_COLOR = "steelblue"
FAMILY = {
    # family      linestyle  hatch  label
    "weight_gd": ("-o", "", "Weight GD"),
    "subspace_gd": (":^", "//", "Subspace GD"),
}
INIT_ORDER = ("pretrained", "avg", "merged", "coeff_best")

# Test-only reference lines pulled from strategies.baselines (see plot helpers
# below). Colours match the corresponding init points for visual coherence.
BASELINE_COLOR = {
    "pretrained": "mediumpurple",
    "merged_avg": "teal",
    "merged_coeff1": "darkorange",
    "upper_bound": "seagreen",
}
BASELINE_LABEL = {
    "pretrained": "Pretrained",
    "merged_avg": "Merged (coeff=1/N)",
    "merged_coeff1": "Merged (coeff=1)",
    "upper_bound": "Multi-task upper bound",
}


@dataclass
class Series:
    family: str            # "coeff_search" | "weight_gd" | "subspace_gd"
    init: Optional[str]    # None for coeff_search
    label: str
    color: str
    style: str             # matplotlib fmt for line plots
    hatch: str             # bar hatch for grouped bars
    hollow: bool = False


# def build_series() -> List[Series]:
#     series = [Series("coeff_search", None, "Coeff search (no GD)", COEFF_COLOR, "--s", "..")]
#     for family, (style, hatch, fam_label) in FAMILY.items():
#         for init in INIT_ORDER:
#             series.append(
#                 Series(family, init, f"{fam_label} ({init})", INIT_COLOR[init], style, hatch)
#             )
#     return series

def _weight_gd_data_init(method: str) -> str:
    """Naive merge init: TA uses avg (1/N); TIES / DARE / TSVM use coeff=1."""
    return "avg" if method == "ta" else "merged"


def build_series(method: str) -> List[Series]:
    sw, hw, _ = FAMILY["weight_gd"]
    ss, hs, _ = FAMILY["subspace_gd"]
    gd_init = _weight_gd_data_init(method)
    return [
        Series("coeff_search", None, "Coeff search (no GD)", COEFF_COLOR, "--s", ".."),
        Series("weight_gd", gd_init, f"Weight GD ({gd_init})", INIT_COLOR[gd_init], sw, hw),
        Series("weight_gd", "coeff_best", "Weight GD (coeff_best)", INIT_COLOR["coeff_best"], sw, hw),
        Series("subspace_gd", "coeff_best", "Subspace GD (coeff_best)", INIT_COLOR["coeff_best"], ss, hs),
    ]


def build_scaling_series(method: str) -> List[Series]:
    """Data-scaling curves: color by strategy; GD from the naive merge only."""
    gd_init = _weight_gd_data_init(method)
    return [
        Series("coeff_search", None, "Coefficient search", STRAT_COLOR["coeff_search"], "--s", ""),
        Series("subspace_gd", "coeff_best", "Subspace GD", STRAT_COLOR["subspace_gd"], ":^", ""),
        Series("weight_gd", gd_init, "GD", STRAT_COLOR["weight_gd"], "-o", ""),
        Series("weight_gd_l2_sp", gd_init, "Regularized GD", STRAT_COLOR["weight_gd_l2_sp"], "-D", "", hollow=True),
    ]


def _pretty_arch(arch: str) -> str:
    parts = arch.replace("_", "-").split("-")
    if parts and parts[0].lower() == "vit":
        return "ViT-" + "-".join(p.upper() for p in parts[1:])
    return arch


def _pretty_method(method: str) -> str:
    return method.upper()


def _budget_tick(budget: str, avg_per_class: float) -> str:
    if budget == "full":
        return f"{avg_per_class:.0f}"
    return budget.replace("per_class", "")


def _as_percent(metric: str) -> bool:
    return "acc" in metric


def series_value(rec: dict, s: Series, metric: str) -> Optional[float]:
    strat = rec.get("strategies", {}).get(s.family)
    if strat is None:
        return None
    node = strat if s.init is None else strat.get(s.init)
    if node is None:
        return None
    return node.get(metric)


def _baselines(rec: dict) -> dict:
    return rec.get("strategies", {}).get("baselines", {})


def _baseline_value(baselines: dict, key: str, metric: str) -> Optional[float]:
    node = baselines.get(key)
    return node.get(metric) if node else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load(paths: List[str]) -> List[dict]:
    records = []
    for path in sorted(paths):
        with open(path) as f:
            rec = json.load(f)
        rec["_path"] = path
        records.append(rec)
    return records


def _avg_per_class(rec: dict) -> float:
    stats = rec.get("data_stats", {})
    if "avg_instances_per_class" in stats:
        return float(stats["avg_instances_per_class"])
    tasks = rec["tasks"]
    return sum(stats[t]["budget_used"] / stats[t]["num_classes"] for t in tasks) / len(tasks)


def _tasks_str(tasks: List[str]) -> str:
    return "-".join(tasks)


def _default_out(results_dir: str, mode: str, tasks: List[str], arch: str, tail: str) -> str:
    return os.path.join(results_dir, PLOTS_SUBDIR, mode, f"{_tasks_str(tasks)}__{arch}__{tail}.png")


def _save(fig, out: str, *, dpi: int = 150, tight: bool = True) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    kwargs = {"dpi": dpi}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(out, **kwargs)
    pdf = os.path.splitext(out)[0] + ".pdf"
    pdf_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    fig.savefig(pdf, **pdf_kwargs)
    plt.close(fig)
    print(f"Saved plot -> {out}")
    print(f"Saved plot -> {pdf}")


# --------------------------------------------------------------------------- #
# Main Result 2: data scaling
# --------------------------------------------------------------------------- #

def _dedupe_seed_records(records: List[dict]) -> List[dict]:
    """One record per (budget, seed); prefer filenames that include __seed."""
    best: Dict[Tuple[str, int], dict] = {}
    for r in records:
        key = (str(r.get("budget", "?")), int(r.get("seed", 42)))
        prev = best.get(key)
        if prev is None or ("__seed" in r.get("_path", "") and "__seed" not in prev.get("_path", "")):
            best[key] = r
    return list(best.values())


def _scaling_panel_data(records: List[dict], metric: str):
    """Return (method, arch, budgets, xs, curves) for one architecture."""
    records = _dedupe_seed_records(records)
    by_budget: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_budget[str(r.get("budget", "?"))].append(r)
    budgets = sorted(by_budget, key=lambda b: _avg_per_class(by_budget[b][0]))
    xs = [_avg_per_class(by_budget[b][0]) for b in budgets]
    scale = 100.0 if _as_percent(metric) else 1.0
    method = records[0]["method"]
    curves = []
    for s in build_scaling_series(method):
        means, stds, xs_plot = [], [], []
        for i, b in enumerate(budgets):
            vals = [v for v in (series_value(r, s, metric) for r in by_budget[b]) if v is not None]
            if not vals:
                continue
            means.append(scale * sum(vals) / len(vals))
            stds.append(0.0 if len(vals) == 1 else scale * statistics.stdev(vals))
            xs_plot.append(i)
        if means:
            curves.append((s, xs_plot, means, stds))
    if not curves:
        raise SystemExit("No strategy series found in the matched result files.")
    return method, records[0]["arch"], budgets, xs, curves


def _draw_scaling_panel(ax, budgets, xs, curves, metric, *, ylabel: bool, title: str,
                       hollow: bool = False) -> None:
    y_lo, y_hi = float("inf"), float("-inf")
    for s, xs_plot, means, stds in curves:
        kw = dict(color=s.color, linewidth=1.6, markersize=5.5, label=s.label, zorder=3)
        if hollow and s.hollow:
            kw.update(markerfacecolor="none", markeredgecolor=s.color, markeredgewidth=1.2)
        ax.plot(xs_plot, means, s.style, **kw)
        lo = [m - d for m, d in zip(means, stds)]
        hi = [m + d for m, d in zip(means, stds)]
        ax.fill_between(xs_plot, lo, hi, color=s.color, alpha=0.12, linewidth=0, zorder=2)
        y_lo, y_hi = min(y_lo, *lo), max(y_hi, *hi)

    ax.set_xlabel("Instances per class", labelpad=3)
    if ylabel:
        ax.set_ylabel("Average accuracy (%)" if _as_percent(metric) else metric)
    ax.set_title(title)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([_budget_tick(b, x) for b, x in zip(budgets, xs)])
    ax.tick_params(axis="x", pad=2)
    ax.set_xlim(-0.4, len(budgets) - 0.6)
    span = max(y_hi - y_lo, 1e-6)
    if _as_percent(metric):
        pad_lo, pad_hi = 1.0, max(0.6, 0.08 * span)
    else:
        pad_lo, pad_hi = 0.01, max(0.006, 0.08 * span)
    ax.set_ylim(y_lo - pad_lo, y_hi + pad_hi)
    if _as_percent(metric):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(False, axis="x")


def plot_data_scaling(groups: List[List[dict]], out: str, metric: str, title: Optional[str]) -> None:
    """One panel per architecture; independent y-limits, one y-label, one legend."""
    panels = [_scaling_panel_data(g, metric) for g in groups]
    n = len(panels)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, n, figsize=SCALE_FIGSIZE if n > 1 else SCALE_FIGSIZE_ONE)
        if n == 1:
            axes = [axes]
        for i, (ax, (method, arch, budgets, xs, curves)) in enumerate(zip(axes, panels)):
            if title and n == 1:
                panel_title = title
            elif n == 1:
                panel_title = f"{_pretty_arch(arch)} ({_pretty_method(method)})"
            else:
                panel_title = _pretty_arch(arch)
            _draw_scaling_panel(ax, budgets, xs, curves, metric, ylabel=(i == 0), title=panel_title)
        handles, labels = max((ax.get_legend_handles_labels() for ax in axes), key=lambda p: len(p[0]))
        ncol = 2 if len(labels) >= 4 else 3
        fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False,
                   handlelength=1.8, handletextpad=0.5, columnspacing=1.6)
        fig.set_layout_engine("constrained")
        if n > 1:
            fig.get_layout_engine().set(wspace=0.08)
        _save(fig, out, dpi=400, tight=False)
    write_scaling_csv(panels, os.path.splitext(out)[0] + ".csv")


def _fmt_pm(mean: float, std: float, *, with_std: bool) -> str:
    if not with_std:
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_scaling_csv(panels, path: str) -> None:
    """Budgets as columns; one row per (arch, series). Mean $\\pm$ std except full."""
    _, _, budgets, xs, _ = panels[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *(_budget_tick(b, x) for b, x in zip(budgets, xs))])
        for _, arch, p_budgets, _, curves in panels:
            idx = {b: i for i, b in enumerate(p_budgets)}
            for s, xs_plot, means, stds in curves:
                by_i = {i: (m, d) for i, m, d in zip(xs_plot, means, stds)}
                row = [f"{_pretty_arch(arch)} {s.label}"]
                for b in budgets:
                    i = idx.get(b)
                    if i not in by_i:
                        row.append("")
                        continue
                    row.append(_fmt_pm(*by_i[i], with_std=(b != "full")))
                w.writerow(row)
    print(f"Saved table -> {path}")


# --------------------------------------------------------------------------- #
# Main Result 1: full comparison
# --------------------------------------------------------------------------- #

def plot_full_comparison(records: List[dict], out: str, metric: str, title: Optional[str]) -> None:
    by_method = {r["method"]: r for r in records}
    methods = sorted(by_method)
    n = len(build_series(methods[0]))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(methods) * 2), 5))
    group_w = 0.8
    bar_w = group_w / n
    base = list(range(len(methods)))

    any_bar = False
    for i in range(n):
        vals, xs, colors, hatches = [], [], [], []
        label = None
        for j, m in enumerate(methods):
            s = build_series(m)[i]
            v = series_value(by_method[m], s, metric)
            if v is None:
                continue
            vals.append(v)
            xs.append(j - group_w / 2 + (i + 0.5) * bar_w)
            colors.append(s.color)
            hatches.append(s.hatch)
            if label is None:
                # One legend entry for the avg/merged slot across methods.
                if s.family == "weight_gd" and s.init in ("avg", "merged"):
                    label = "Weight GD (avg/merged)"
                else:
                    label = s.label
        if not vals:
            continue
        ax.bar(xs, vals, bar_w, label=label, color=colors, hatch=hatches[0],
               edgecolor="white", linewidth=0.4)
        any_bar = True

    if not any_bar:
        raise SystemExit("No strategy series found in the matched result files.")

    # pretrained and upper_bound are method-independent -> global axhlines.
    global_baselines = next((_baselines(r) for r in records if _baselines(r)), {})
    for key in ("pretrained", "upper_bound"):
        v = _baseline_value(global_baselines, key, metric)
        if v is not None:
            ax.axhline(v, color=BASELINE_COLOR[key], linestyle="--", linewidth=1.5,
                       alpha=0.85, label=f"{BASELINE_LABEL[key]} = {v:.4f}")

    # merged_avg and merged_coeff1 
    # horizontal tick above each method's bar group.
    for key in ("merged_avg", "merged_coeff1"):
        added_label = False
        for j, m in enumerate(methods):
            v = _baseline_value(_baselines(by_method[m]), key, metric)
            if v is None:
                continue
            left = j - group_w / 2
            right = j + group_w / 2
            ax.hlines(v, left, right, colors=BASELINE_COLOR[key],
                      linestyles="--", linewidth=2,
                      label=(BASELINE_LABEL[key] if not added_label else None))
            added_label = True

    ax.set_xticks(base)
    ax.set_xticklabels(methods)
    ax.set_xlabel("Merging method")
    ax.set_ylabel(f"{metric}")
    tasks = records[0]["tasks"]
    budget = records[0].get("budget", "?")
    ax.set_title(title or f"Full comparison - {', '.join(tasks)} ({records[0]['arch']}, budget={budget})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _tasks(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _unit_prefix(tasks: List[str]) -> str:
    return f"{len(tasks)}_tasks_{_tasks_str(tasks)}"


def _attach_l2_sp_weight_gd(main_records: List[dict], l2_sp_records: List[dict]) -> List[dict]:
    """Copy weight_gd from an L2-SP run onto main-exp records as weight_gd_l2_sp."""
    def key(r):
        return (str(r.get("budget")), int(r.get("seed", 42)))

    by_l2_sp = {key(r): r for r in l2_sp_records}
    for rec in main_records:
        src = by_l2_sp.get(key(rec))
        wg = (src or {}).get("strategies", {}).get("weight_gd")
        if wg is not None:
            rec.setdefault("strategies", {})["weight_gd_l2_sp"] = wg
    return main_records


def _run_data_scaling(args) -> None:
    tasks = _tasks(args.tasks)
    arches = _tasks(args.arch)
    groups = []
    for arch in arches:
        pat = os.path.join(args.results_dir, UNITS_SUBDIR,
                           f"{_unit_prefix(tasks)}__{arch}__{args.method}__*.json")
        paths = glob.glob(pat)
        if not paths:
            raise SystemExit(f"No result files matching {pat}")
        recs = _load(paths)
        if getattr(args, "l2_sp_results_dir", None):
            l2_sp_pat = os.path.join(args.l2_sp_results_dir, UNITS_SUBDIR,
                                     f"{_unit_prefix(tasks)}__{arch}__{args.method}__*.json")
            l2_sp_paths = glob.glob(l2_sp_pat)
            if not l2_sp_paths:
                raise SystemExit(f"No L2-SP result files matching {l2_sp_pat}")
            recs = _attach_l2_sp_weight_gd(recs, _load(l2_sp_paths))
        groups.append(recs)
    arch_tag = arches[0] if len(arches) == 1 else "-".join(arches)
    out_root = args.l2_sp_results_dir or args.results_dir
    out = args.out or _default_out(out_root, "data_scaling", tasks, arch_tag, args.method)
    plot_data_scaling(groups, out, args.metric, args.title)


def _run_full_comparison(args) -> None:
    tasks = _tasks(args.tasks)
    pat = os.path.join(args.results_dir, UNITS_SUBDIR,
                       f"{_unit_prefix(tasks)}__{args.arch}__*__{args.budget}.json")
    paths = glob.glob(pat)
    if not paths:
        raise SystemExit(f"No result files matching {pat}")
    out = args.out or _default_out(args.results_dir, "full_comparison", tasks, args.arch, args.budget)
    plot_full_comparison(_load(paths), out, args.metric, args.title)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def _common(sp):
        sp.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Results root; reads units/ and writes plots/ under it")
        sp.add_argument("--tasks", required=True, help="Comma-separated task labels (order must match the runs)")
        sp.add_argument("--arch", required=True,
                        help="Arch tag, or comma-separated pair for side-by-side (e.g. vit-b-32,vit-b-16)")
        sp.add_argument("--metric", default=DEFAULT_METRIC,
                        help="Metric key to plot (test_avg_acc | valid_avg_acc | test_avg_loss | ...)")
        sp.add_argument("--out", default=None, help="Output PNG path (default: plots/<mode>/...)")
        sp.add_argument("--title", default=None)

    sp_scale = sub.add_parser("data_scaling", help="Accuracy vs valid budget (Main Result 2)")
    _common(sp_scale)
    sp_scale.add_argument("--method", required=True, help="Merging method, e.g. ta")
    sp_scale.add_argument("--l2-sp-results-dir", default=None,
                         help="Optional L2-SP results root; overlays weight_gd as Regularized GD")
    sp_scale.set_defaults(func=_run_data_scaling)

    sp_cmp = sub.add_parser("full_comparison", help="Strategy x method grouped bars (Main Result 1)")
    _common(sp_cmp)
    sp_cmp.add_argument("--budget", default="full", help="Budget label to compare across methods (default: full)")
    sp_cmp.set_defaults(func=_run_full_comparison)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
