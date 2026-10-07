#!/usr/bin/env python3
"""Build weight-GD learning-rate tables from weight_gd_lr unit JSONs.

Writes two CSVs of test average accuracy (%):
  * full budget — TA and TIES, both arches
  * 1/4/16 per class — TA only (TIES was not run at those budgets)

Validation accuracy saturates (~100%), so * marks the highest test cell
in each (arch, method) group (full) or (arch, budget) group (scaling).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

VISION_EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = VISION_EXP_ROOT / "results" / "weight_gd_lr"
DEFAULT_LR = 1e-5

ARCH_ORDER = ("vit-b-32", "vit-b-16")
METHOD_ORDER = ("ta", "ties")
INIT_ORDER = ("avg", "coeff_best")
SCALE_BUDGETS = ("1per_class", "4per_class", "16per_class")
METHOD_LABEL = {"ta": "Task Arithmetic", "ties": "TIES"}
BUDGET_LABEL = {
    "1per_class": "1 / class",
    "4per_class": "4 / class",
    "16per_class": "16 / class",
}


def _pretty_arch(arch: str) -> str:
    parts = arch.replace("_", "-").split("-")
    if parts and parts[0].lower() == "vit":
        return "ViT-" + "-".join(p.upper() for p in parts[1:])
    return arch


def _fmt_lr(lr: float) -> str:
    if math.isclose(lr, DEFAULT_LR):
        return "1e-5 (def.)"
    return f"{lr:.0e}".replace("e-0", "e-")


def _fmt_pct(acc: float, starred: bool) -> str:
    text = f"{100 * acc:.2f}"
    return text + "*" if starred else text


def _star(cells: dict, group_fn) -> set:
    """Keys with the highest test acc inside each group_fn(key) bucket."""
    best = {}
    for key, acc in cells.items():
        group = group_fn(key)
        if group not in best or acc > cells[best[group]]:
            best[group] = key
    return set(best.values())


def load_cells(root: Path) -> dict[tuple, float]:
    """Map (budget, lr, arch, method, init) -> test_avg_acc."""
    cells = {}
    for path in sorted(root.glob("lr*/units/*.json")):
        rec = json.loads(path.read_text())
        weight_gd = rec.get("strategies", {}).get("weight_gd")
        if not weight_gd:
            continue
        cfg = rec["config"]
        lr = float(cfg["gd_lr"])
        key_prefix = (cfg["budget"], lr, cfg["arch"], cfg["method"])
        for init, node in weight_gd.items():
            cells[(*key_prefix, init)] = node["test_avg_acc"]
    if not cells:
        raise SystemExit(f"no weight_gd cells under {root}")
    return cells


def write_full(cells: dict, out: Path) -> None:
    cols = [(a, m, i) for a in ARCH_ORDER for m in METHOD_ORDER for i in INIT_ORDER]
    lrs = sorted({lr for _, lr, *_ in cells})
    starred = _star(cells, lambda k: (k[2], k[3]))  # (arch, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *(_pretty_arch(a) for a, _, _ in cols)])
        w.writerow(["", *(METHOD_LABEL[m] for _, m, _ in cols)])
        w.writerow(["LR", *(init for _, _, init in cols)])
        for lr in lrs:
            row = [_fmt_lr(lr)]
            for col in cols:
                key = ("full", lr, *col)
                row.append(_fmt_pct(cells[key], key in starred) if key in cells else "")
            w.writerow(row)


def write_scaling(cells: dict, out: Path) -> None:
    cols = [(a, i) for a in ARCH_ORDER for i in INIT_ORDER]
    lrs = sorted({lr for _, lr, *_ in cells})
    starred = _star(cells, lambda k: (k[0], k[2]))  # (budget, arch)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "", *(_pretty_arch(a) for a, _ in cols)])
        w.writerow(["Budget", "LR", *(init for _, init in cols)])
        for budget in SCALE_BUDGETS:
            for i, lr in enumerate(lrs):
                row = [BUDGET_LABEL[budget] if i == 0 else "", _fmt_lr(lr)]
                for arch, init in cols:
                    key = (budget, lr, arch, "ta", init)
                    row.append(_fmt_pct(cells[key], key in starred) if key in cells else "")
                w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    cells = load_cells(args.results_dir)
    full = {k: v for k, v in cells.items() if k[0] == "full"}
    scale = {k: v for k, v in cells.items() if k[0] in SCALE_BUDGETS}
    if not full:
        raise SystemExit("no full-budget weight_gd cells")
    if not scale:
        raise SystemExit("no 1/4/16-per-class weight_gd cells")
    full_out = args.results_dir / "weight_gd_lr_table.csv"
    scale_out = args.results_dir / "weight_gd_lr_scaling_table.csv"
    write_full(full, full_out)
    write_scaling(scale, scale_out)
    print(f"wrote {full_out}")
    print(f"wrote {scale_out}")


if __name__ == "__main__":
    main()
