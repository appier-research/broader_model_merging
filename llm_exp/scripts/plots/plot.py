#!/usr/bin/env python3
"""Plot the merge-experiment results produced by scripts/merge_eval.py.

Results live as one JSON per atomic unit (see src/experiment.py):

    <results-dir>/units/<n>_tasks_<tasks>__<arch>__<method>__<budget>.json

and figures are written alongside them under <results-dir>/plots/<mode>/.
Same unit-JSON-glob pattern as vision_exp/scripts/plots/plot.py; the only
substantive differences are the metric names (LLM tasks score generations,
not classification accuracy: ``test_avg_score``/``valid_avg_score`` instead
of ``*_acc``) and the budget axis (LLM budgets are the discrete labels
low/medium/full, not a per-class integer sweep -- see src/data.py).

Two plot modes mirror the two main results:

  * budget_scaling    -- avg score vs valid budget, one curve per
                         strategy/init, for a fixed (tasks, arch, method).
  * full_comparison    -- grouped bars of strategy/init score across
                         merging methods, for a fixed (tasks, arch, budget).

Both modes glob the results directory, so a plot always reflects every unit
that has been computed so far (e.g. rerun full_comparison after adding TIES
to include it automatically). Missing strategy/init cells are simply
skipped.

Examples
--------
    python scripts/plots/plot.py budget_scaling --tasks bank77,ddxplus,ifeval,usefulness_judge \\
        --arch qwen3-0.6b --method ta
    python scripts/plots/plot.py full_comparison --tasks bank77,ddxplus,ifeval,usefulness_judge \\
        --arch qwen3-0.6b --budget full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_METRIC = "test_avg_score"

# Subfolders inside the results root (mirrors src/experiment.py):
#   units/  per-unit JSONs we read;  plots/  figures we write.
UNITS_SUBDIR = "units"
PLOTS_SUBDIR = "plots"

# One consistent colour per init point, one linestyle/hatch per strategy family.
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


def build_series() -> List[Series]:
    series = [Series("coeff_search", None, "Coeff search (no GD)", COEFF_COLOR, "--s", "..")]
    for family, (style, hatch, fam_label) in FAMILY.items():
        for init in INIT_ORDER:
            series.append(
                Series(family, init, f"{fam_label} ({init})", INIT_COLOR[init], style, hatch)
            )
    return series


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


def _full_baselines(records: List[dict]) -> dict:
    """Return the baselines dict from the 'full' budget sibling, if any."""
    for r in records:
        if r.get("budget") == "full" and _baselines(r):
            return _baselines(r)
    return {}


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


def _total_budget(rec: dict) -> int:
    return int(rec.get("data_stats", {}).get("total_budget_used", 0))


def _tasks_str(tasks: List[str]) -> str:
    return "-".join(tasks)


def _default_out(results_dir: str, mode: str, tasks: List[str], arch: str, tail: str) -> str:
    return os.path.join(results_dir, PLOTS_SUBDIR, mode, f"{_tasks_str(tasks)}__{arch}__{tail}.png")


def _save(fig, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out}")


# --------------------------------------------------------------------------- #
# Main Result 2: budget scaling
# --------------------------------------------------------------------------- #

def plot_budget_scaling(records: List[dict], out: str, metric: str, title: Optional[str]) -> None:
    records = sorted(records, key=_total_budget)
    x = list(range(len(records)))
    x_labels = [str(r.get("budget", "?")) for r in records]
    x_budgets = [_total_budget(r) for r in records]

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for s in build_series():
        pairs = [(xi, series_value(r, s, metric)) for xi, r in zip(x, records)]
        pairs = [(xi, yi) for xi, yi in pairs if yi is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, s.style, color=s.color, linewidth=2, markersize=7, label=s.label)
        plotted += 1

    if plotted == 0:
        raise SystemExit("No strategy series found in the matched result files.")

    for key in ("pretrained", "merged_avg", "merged_coeff1", "upper_bound"):
        v = _baseline_value(_full_baselines(records), key, metric)
        if v is not None:
            ax.axhline(v, color=BASELINE_COLOR[key], linestyle="--", linewidth=1.5,
                       alpha=0.85, label=f"{BASELINE_LABEL[key]} = {v:.4f}")

    ax.set_xlabel("Valid budget")
    ax.set_ylabel(f"{metric}")
    tasks = records[0]["tasks"]
    ax.set_title(title or f"Budget scaling - {', '.join(tasks)} ({records[0]['arch']}, {records[0]['method']})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({v} total)" for m, v in zip(x_labels, x_budgets)], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out)


# --------------------------------------------------------------------------- #
# Main Result 1: full comparison
# --------------------------------------------------------------------------- #

def plot_full_comparison(records: List[dict], out: str, metric: str, title: Optional[str]) -> None:
    by_method = {r["method"]: r for r in records}
    methods = sorted(by_method)
    series = build_series()

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(methods) * 2), 5))
    n = len(series)
    group_w = 0.8
    bar_w = group_w / n
    base = list(range(len(methods)))

    any_bar = False
    for i, s in enumerate(series):
        vals, xs = [], []
        for j, m in enumerate(methods):
            v = series_value(by_method[m], s, metric)
            if v is None:
                continue
            vals.append(v)
            xs.append(j - group_w / 2 + (i + 0.5) * bar_w)
        if not vals:
            continue
        ax.bar(xs, vals, bar_w, label=s.label, color=s.color, hatch=s.hatch,
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

    # merged_avg and merged_coeff1 get a per-method horizontal tick above each
    # method's bar group (they depend on the method's own task vectors).
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


def _run_budget_scaling(args) -> None:
    tasks = _tasks(args.tasks)
    pat = os.path.join(args.results_dir, UNITS_SUBDIR,
                       f"{_unit_prefix(tasks)}__{args.arch}__{args.method}__*.json")
    paths = glob.glob(pat)
    if not paths:
        raise SystemExit(f"No result files matching {pat}")
    out = args.out or _default_out(args.results_dir, "budget_scaling", tasks, args.arch, args.method)
    plot_budget_scaling(_load(paths), out, args.metric, args.title)


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
        sp.add_argument("--results-dir", default="results",
                        help="Results root; reads units/ and writes plots/ under it")
        sp.add_argument("--tasks", required=True, help="Comma-separated task labels (order must match the runs)")
        sp.add_argument("--arch", required=True, help="Arch tag, e.g. qwen3-0.6b")
        sp.add_argument("--metric", default=DEFAULT_METRIC,
                        help="Metric key to plot (test_avg_score | valid_avg_score | ...)")
        sp.add_argument("--out", default=None, help="Output PNG path (default: plots/<mode>/...)")
        sp.add_argument("--title", default=None)

    sp_scale = sub.add_parser("budget_scaling", help="Score vs valid budget (Main Result 2)")
    _common(sp_scale)
    sp_scale.add_argument("--method", required=True, help="Merging method, e.g. ta")
    sp_scale.set_defaults(func=_run_budget_scaling)

    sp_cmp = sub.add_parser("full_comparison", help="Strategy x method grouped bars (Main Result 1)")
    _common(sp_cmp)
    sp_cmp.add_argument("--budget", default="full", help="Budget label to compare across methods (default: full)")
    sp_cmp.set_defaults(func=_run_full_comparison)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
