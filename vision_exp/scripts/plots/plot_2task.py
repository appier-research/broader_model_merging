#!/usr/bin/env python3
"""2-task data-scaling figures (not the 9-task plot.py layout).

Writes two figures under the L2-SP (or main) results plots/ folder:
  1. Same single-panel test curve as the 9-task subfigure, title includes tasks
  2. Train | Test side-by-side, one shared legend row
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot as P

TASK_LABEL = {
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "fer2013": "FER2013",
    "food101": "Food-101",
    "gtsrb": "GTSRB",
    "mnist": "MNIST",
    "resisc45": "RESISC45",
    "stanford-cars": "Stanford Cars",
    "sun397": "SUN397",
}


def _pretty_tasks(tasks: list[str]) -> str:
    return " + ".join(TASK_LABEL.get(t, t.replace("-", " ").title()) for t in tasks)


def _load_groups(args) -> tuple[list[str], list[list[dict]]]:
    tasks = P._tasks(args.tasks)
    if len(tasks) != 2:
        raise SystemExit(f"plot_2task expects exactly 2 tasks, got {tasks}")
    groups = []
    for arch in P._tasks(args.arch):
        pat = os.path.join(args.results_dir, P.UNITS_SUBDIR,
                           f"{P._unit_prefix(tasks)}__{arch}__{args.method}__*.json")
        paths = glob.glob(pat)
        if not paths:
            raise SystemExit(f"No result files matching {pat}")
        recs = P._load(paths)
        if args.l2_sp_results_dir:
            l2_pat = os.path.join(args.l2_sp_results_dir, P.UNITS_SUBDIR,
                                  f"{P._unit_prefix(tasks)}__{arch}__{args.method}__*.json")
            l2_paths = glob.glob(l2_pat)
            if not l2_paths:
                raise SystemExit(f"No L2-SP result files matching {l2_pat}")
            recs = P._attach_l2_sp_weight_gd(recs, P._load(l2_paths))
        groups.append(recs)
    return tasks, groups


def _fig_title(tasks: list[str], arch: str, method: str) -> str:
    return f"{_pretty_tasks(tasks)} — {P._pretty_arch(arch)} ({P._pretty_method(method)})"


def plot_train_test(records: list[dict], out: str, title: str) -> None:
    """Left: valid_avg_acc (train set). Right: test_avg_acc. Shared one-row legend."""
    train = P._scaling_panel_data(records, "valid_avg_acc")
    test = P._scaling_panel_data(records, "test_avg_acc")
    with plt.rc_context(P.PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=P.SCALE_FIGSIZE)
        fig.suptitle(title)
        for ax, panel, name, ylabel in (
            (axes[0], train, "Train", True),
            (axes[1], test, "Test", False),
        ):
            _, _, budgets, xs, curves = panel
            P._draw_scaling_panel(ax, budgets, xs, curves, "acc", ylabel=ylabel, title=name,
                                 hollow=(name == "Train"))
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
                   handlelength=1.8, handletextpad=0.5, columnspacing=1.6)
        fig.set_layout_engine("constrained")
        fig.get_layout_engine().set(wspace=0.08)
        P._save(fig, out, dpi=400, tight=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=P.DEFAULT_RESULTS_DIR)
    p.add_argument("--tasks", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--l2-sp-results-dir", default=None)
    p.add_argument("--metric", default=P.DEFAULT_METRIC,
                   help="Metric for the single-panel figure (default: test_avg_acc)")
    args = p.parse_args()

    tasks, groups = _load_groups(args)
    arch = P._tasks(args.arch)[0]
    out_root = args.l2_sp_results_dir or args.results_dir
    stem = P._default_out(out_root, "data_scaling", tasks, arch, args.method)
    title = _fig_title(tasks, arch, args.method)

    P.plot_data_scaling(groups, stem, args.metric, title)
    plot_train_test(groups[0], os.path.splitext(stem)[0] + "__train_test.png", title)


if __name__ == "__main__":
    main()
