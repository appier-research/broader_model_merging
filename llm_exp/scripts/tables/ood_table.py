#!/usr/bin/env python3
"""Tabulate the OOD-generalization units: seen vs unseen (gsm8k / mbpp) per cell.

Reads every ``*__unseen_*__*.json`` under ``<root>/units/`` (default root
results/ood_generalization -- written by scripts/ood_eval.py or by
scripts/merge_eval.py --unseen-tasks) and writes next to them:

  summary.csv   one row per (variant, unit, cell): seen avg, unseen avg, per-task unseen
  tables.md     one markdown table per (arch, unseen split), methods x cells

``--variant label=path`` adds another results root (e.g. a rerun with a
different training budget) whose rows appear next to the main root's, tagged
with ``label``; the main root's rows are tagged ``main``.

Rows follow the paper's row order (pretrained, naive merge, coeff search, GD
inits...) and only cells that actually carry unseen numbers are listed.
LLM counterpart of vision_exp/scripts/tables/ood_table.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELL_ORDER = [
    ("baselines/pretrained", "Pretrained"),
    ("baselines/merged_avg", "merge coeff=1/N"),
    ("baselines/merged_coeff1", "merge coeff=1"),
    ("baselines/upper_bound", "Multi-task (UB)"),
    ("coeff_search", "coeff search"),
    ("weight_gd/pretrained", "weight GD (init pretrained)"),
    ("weight_gd/avg", "weight GD (init 1/N)"),
    ("weight_gd/merged", "weight GD (init merged)"),
    ("weight_gd/coeff_best", "weight GD (init coeff*)"),
    ("weight_gd_lora/pretrained", "LoRA GD (init pretrained)"),
    ("weight_gd_lora/avg", "LoRA GD (init 1/N)"),
    ("weight_gd_lora/merged", "LoRA GD (init merged)"),
    ("weight_gd_lora/coeff_best", "LoRA GD (init coeff*)"),
    ("subspace_gd/pretrained", "subspace GD (init pretrained)"),
    ("subspace_gd/avg", "subspace GD (init 1/N)"),
    ("subspace_gd/merged", "subspace GD (init merged)"),
    ("subspace_gd/coeff_best", "subspace GD (init coeff*)"),
    ("bo_search/coeff_best", "BO (center coeff*)"),
]
_LABEL = dict(CELL_ORDER)
_RANK = {k: i for i, (k, _) in enumerate(CELL_ORDER)}


def pct(x) -> str:
    return "" if x is None else f"{100.0 * float(x):.1f}"


def cells(strategies: dict):
    out = []
    for strat, block in strategies.items():
        if strat == "coeff_search":
            out.append((strat, block))
        elif isinstance(block, dict):
            for init, cell in block.items():
                if isinstance(cell, dict):
                    out.append((f"{strat}/{init}", cell))
    rows = []
    for p, c in out:
        if "unseen_test_per_task" in c:
            rows.append((p, c))
        if "unseen_raw_test_per_task" in c:  # pretrained *-Base reference, raw-completion prompts
            rows.append((p + " [raw]", {
                **{k: v for k, v in c.items() if not k.startswith("unseen_")},
                "unseen_test_per_task": c["unseen_raw_test_per_task"],
                "unseen_test_avg_score": c["unseen_raw_test_avg_score"],
                "unseen_source": "raw-" + c.get("unseen_raw_source", ""),
            }))
    return sorted(rows, key=lambda pc: (_RANK.get(pc[0].replace(" [raw]", ""), 999), pc[0]))


def load_units(units_dir: Path):
    for path in sorted(units_dir.glob("*__unseen_*.json")):
        d = json.loads(path.read_text())
        yield path.name, d


def _rows(variant: str, fname: str, d: dict) -> list:
    unseen = d.get("unseen_tasks") or []
    out = []
    for path, cell in cells(d.get("strategies", {})):
        per = cell["unseen_test_per_task"]
        out.append({
            "variant": variant, "unit": fname, "arch": d["arch"], "method": d["method"],
            "unseen_split": "-".join(unseen), "cell": path, "label": _LABEL.get(path, path),
            "seen_avg": cell.get("test_avg_score"), "unseen_avg": cell.get("unseen_test_avg_score"),
            "per_task": {t: per.get(t) for t in unseen},
            "source": cell.get("unseen_source", "live"),
            "epochs": cell.get("epochs"),
            "shots": "/".join(str((d.get("unseen_n_shot") or {}).get(t, 0)) for t in unseen),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "results" / "ood_generalization"))
    p.add_argument("--variant", action="append", default=[], metavar="LABEL=PATH",
                   help="Extra results root to tabulate alongside --root (repeatable)")
    args = p.parse_args()
    root = Path(args.root)
    roots = [("main", root)]
    for v in args.variant:
        if "=" not in v:
            p.error(f"--variant expects LABEL=PATH, got {v!r}")
        label, path = v.split("=", 1)
        roots.append((label, Path(path)))
    if not (root / "units").is_dir():
        raise SystemExit(f"missing {root / 'units'}")

    rows = []
    for variant, vroot in roots:
        units_dir = vroot / "units"
        if not units_dir.is_dir():
            print(f"warning: no units under {units_dir}, skipping variant {variant}")
            continue
        for fname, d in load_units(units_dir):
            rows.extend(_rows(variant, fname, d))
    if not rows:
        raise SystemExit(f"no cells with unseen results under {[str(r) for _, r in roots]}")

    all_unseen = sorted({t for r in rows for t in r["per_task"]})
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "arch", "method", "unseen_split", "shots", "cell", "label", "epochs", "seen_avg",
                    "unseen_avg", *all_unseen, "source", "unit"])
        for r in rows:
            w.writerow([
                r["variant"], r["arch"], r["method"], r["unseen_split"], r["shots"], r["cell"], r["label"], r["epochs"] or "",
                pct(r["seen_avg"]), pct(r["unseen_avg"]), *[pct(r["per_task"].get(t)) for t in all_unseen],
                r["source"], r["unit"],
            ])
    print(f"wrote {root / 'summary.csv'}")

    def md_row(vals):
        return "| " + " | ".join(str(v) for v in vals) + " |"

    lines = [
        "# OOD generalization (LLM)", "",
        "Score (%). Seen = average over the 4 merged tasks' test pools (copied from the main unit);",
        "unseen = held-out tasks with no task vector, scored on each cell's final weights.",
        "`source` says how the weights were obtained: live (scored during the run), checkpoint,",
        "scalar (merge rebuilt from task vectors), coeffs (basis x coefficients_final), base.",
        "`[raw]` = pretrained scored with raw-completion few-shot prompts (no chat template).",
        "Regenerate: `python scripts/tables/ood_table.py`.", "",
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arch"], r["unseen_split"], r["shots"])].append(r)
    for (arch, split, shots), grp in sorted(groups.items()):
        unseen = split.split("-")
        lines += [f"## {arch} -- unseen: {', '.join(unseen)} ({shots} shots)", ""]
        methods = sorted({r["method"] for r in grp})
        multi = len({r["variant"] for r in grp}) > 1
        hdr = ["Method", "Cell", *(["Variant"] if multi else []), "Seen avg", *unseen, "Unseen avg", "source"]
        lines += [md_row(hdr), md_row(["---"] * len(hdr))]
        for m in methods:
            sub = [x for x in grp if x["method"] == m]
            sub.sort(key=lambda r: (_RANK.get(r["cell"], 999), r["cell"], r["variant"] != "main", r["variant"]))
            for r in sub:
                lines.append(md_row([
                    m.upper(), r["label"], *([r["variant"]] if multi else []), pct(r["seen_avg"]),
                    *[pct(r["per_task"].get(t)) for t in unseen], pct(r["unseen_avg"]), r["source"],
                ]))
        lines.append("")
    (root / "tables.md").write_text("\n".join(lines))
    print(f"wrote {root / 'tables.md'}")


if __name__ == "__main__":
    main()
