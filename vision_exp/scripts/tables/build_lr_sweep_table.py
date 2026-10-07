#!/usr/bin/env python3
"""Build the subspace-GD learning-rate table from lr_sweep unit JSONs.

One row per LR; columns are Task Arithmetic (avg, coeff_best) and TIES
(merged, coeff_best). Cells are test average accuracy (%). A trailing *
marks the (lr, init) with the highest validation accuracy per method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

VISION_EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITS = VISION_EXP_ROOT / "results" / "lr_sweep" / "units"
DEFAULT_OUT = VISION_EXP_ROOT / "results" / "lr_sweep" / "subspace_gd_lr_table.csv"
DEFAULT_LR = 1e-2

# column order matches the appendix table
COLUMNS = (
    ("ta", "avg"),
    ("ta", "coeff_best"),
    ("ties", "merged"),
    ("ties", "coeff_best"),
)
METHOD_LABEL = {"ta": "Task Arithmetic", "ties": "TIES"}


def _fmt_lr(lr: float) -> str:
    if math.isclose(lr, DEFAULT_LR):
        return "1e-2 (def.)"
    if lr >= 1:
        return f"{lr:g}"
    return f"{lr:.0e}".replace("e-0", "e-")


def _fmt_pct(acc: float, starred: bool) -> str:
    text = f"{100 * acc:.2f}"
    return text + "*" if starred else text


def load_cells(units_dir: Path) -> dict[tuple[float, str, str], tuple[float, float]]:
    """Map (lr, method, init) -> (valid_avg_acc, test_avg_acc)."""
    cells = {}
    for path in sorted(units_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        lr = float(rec["config"]["subspace_lr"])
        method = rec["config"]["method"]
        for init, node in rec["strategies"]["subspace_gd"].items():
            cells[(lr, method, init)] = (node["valid_avg_acc"], node["test_avg_acc"])
    if not cells:
        raise SystemExit(f"no unit JSONs in {units_dir}")
    return cells


def selected_keys(cells: dict) -> set[tuple[float, str, str]]:
    """One (lr, method, init) per method: highest validation accuracy."""
    best: dict[str, tuple[float, str, str]] = {}
    for key, (valid, _) in cells.items():
        method = key[1]
        if method not in best or valid > cells[best[method]][0]:
            best[method] = key
    return set(best.values())


def write_table(cells, out: Path) -> None:
    lrs = sorted({lr for lr, _, _ in cells})
    starred = selected_keys(cells)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *(METHOD_LABEL[m] for m, _ in COLUMNS)])
        writer.writerow(["LR", *(init for _, init in COLUMNS)])
        for lr in lrs:
            row = [_fmt_lr(lr)]
            for method, init in COLUMNS:
                key = (lr, method, init)
                if key not in cells:
                    row.append("")
                    continue
                row.append(_fmt_pct(cells[key][1], key in starred))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units-dir", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    write_table(load_cells(args.units_dir), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
