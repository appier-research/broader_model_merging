#!/usr/bin/env python3
"""Build the Weight-GD SGD vs AdamW table (ViT-B/32, seed 42).

Rows: coefficient search and subspace GD (coeff_best) as controls, then
Weight GD with AdamW (main_exp) and SGD (weight_gd_sgd/m0.0), plus
SGD − AdamW. Weight GD uses the data-scaling init: TA avg, TIES merged.
Columns are TA budgets plus TIES at full only. Cells are test average
accuracy (%); the last row is the signed difference.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VISION_EXP_ROOT = Path(__file__).resolve().parents[2]
UNIT_PREFIX = (
    "9_tasks_dtd-eurosat-fer2013-food101-gtsrb-mnist-resisc45-stanford-cars-sun397"
    "__vit-b-32"
)
TA_BUDGETS = (
    "1per_class",
    "2per_class",
    "4per_class",
    "8per_class",
    "16per_class",
    "32per_class",
    "64per_class",
    "full",
)
COLUMNS = [("ta", b) for b in TA_BUDGETS] + [("ties", "full")]
BUDGET_LABEL = {
    "1per_class": "1 / class",
    "2per_class": "2 / class",
    "4per_class": "4 / class",
    "8per_class": "8 / class",
    "16per_class": "16 / class",
    "32per_class": "32 / class",
    "64per_class": "64 / class",
    "full": "full",
}
METHOD_LABEL = {"ta": "Task Arithmetic", "ties": "TIES"}
ROWS = (
    ("Coefficient search", "coeff_search", None, "adam"),
    ("Subspace GD", "subspace_gd", "coeff_best", "adam"),
    ("Weight GD (AdamW)", "weight_gd", None, "adam"),
    ("Weight GD (SGD)", "weight_gd", None, "sgd"),
)


def _unit_path(root: Path, method: str, budget: str, seeded: bool) -> Path:
    name = f"{UNIT_PREFIX}__{method}__{budget}"
    if seeded and budget != "full":
        name += "__seed42"
    return root / f"{name}.json"


def _gd_init(method: str) -> str:
    return "avg" if method == "ta" else "merged"


def _test_acc(rec: dict, family: str, init: str | None, method: str) -> float:
    node = rec["strategies"][family]
    if family == "weight_gd":
        init = _gd_init(method)
    if init is not None:
        node = node[init]
    return 100 * node["test_avg_acc"]


def load_units(root: Path, seeded: bool) -> dict[tuple[str, str], dict]:
    units = {}
    for method, budget in COLUMNS:
        path = _unit_path(root, method, budget, seeded)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        units[(method, budget)] = json.loads(path.read_text())
    return units


def write_table(adam: dict, sgd: dict, out: Path) -> None:
    sources = {"adam": adam, "sgd": sgd}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *(METHOD_LABEL[m] for m, _ in COLUMNS)])
        w.writerow(["", *(BUDGET_LABEL[b] for _, b in COLUMNS)])
        gd = {}
        for label, family, init, src in ROWS:
            row = [label]
            for method, budget in COLUMNS:
                rec = sources[src][(method, budget)]
                cell = f"{_test_acc(rec, family, init, method):.2f}"
                row.append(cell)
                if family == "weight_gd":
                    gd[(src, method, budget)] = float(cell)
            w.writerow(row)
        delta = ["SGD − AdamW"]
        for method, budget in COLUMNS:
            delta.append(f"{gd[('sgd', method, budget)] - gd[('adam', method, budget)]:+.2f}")
        w.writerow(delta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adam-dir",
        type=Path,
        default=VISION_EXP_ROOT / "results" / "main_exp" / "units",
    )
    parser.add_argument(
        "--sgd-dir",
        type=Path,
        default=VISION_EXP_ROOT / "results" / "weight_gd_sgd" / "m0.0" / "units",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=VISION_EXP_ROOT / "results" / "weight_gd_sgd" / "weight_gd_sgd_table.csv",
    )
    args = parser.parse_args()
    write_table(load_units(args.adam_dir, seeded=True), load_units(args.sgd_dir, seeded=False), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
