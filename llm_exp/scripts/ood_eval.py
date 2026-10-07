#!/usr/bin/env python3
"""Post-hoc out-of-domain eval of an existing unit: no GD rerun.

`scripts/merge_eval.py --unseen-tasks ...` scores held-out tasks *while* a unit runs
(vision_exp's runs/ood_generalization protocol). For the LLM units that
already exist that would mean re-running hours of weight_gd / subspace_gd
just to add two numbers per cell. Every cell of a finished unit is instead
reconstructible without training:

  * baselines/pretrained                -> the base model
  * baselines/merged_avg, merged_coeff1 -> method.merged_delta(coeff) from the task vectors
  * coeff_search                        -> merged_delta(best_lambda)
  * weight_gd/*, weight_gd_lora/*       -> the saved checkpoint (checkpoints/finetuned/...,
                                           admin/download_finetuned.sh pulls them from GCS)
  * subspace_gd/*, bo_search/*          -> saved checkpoint if present, else
                                           base + sum_i c_i dir_i from coefficients_final
  * baselines/upper_bound               -> its checkpoint
  * directional_sampling                -> skipped (ablation; its top-1 draw isn't stored)

Each cell's weights are loaded into one resident model and scored on the
unseen tasks; the result is the source unit copied to
``<out-dir>/units/<name>__unseen_<tasks>__...json`` with
``unseen_test_avg_score`` / ``unseen_test_per_task`` added per cell -- the
same fields `--unseen-tasks` writes live, so scripts/tables/ood_table.py reads both.
Seen numbers are copied verbatim, never recomputed. Incremental: cells that
already carry every requested unseen task are skipped unless --force; the
JSON is saved after every cell.

Example:
  python scripts/ood_eval.py \\
      --unit results/units/4_tasks_bank77-ddxplus-ifeval-usefulness_judge__qwen3-0.6b__ta__full.json \\
      --unseen-tasks gsm8k,mbpp --unseen-test-samples 300
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src import data, models  # noqa: E402
from src.checkpoints import _dir_with_weights, resolve_task_checkpoints  # noqa: E402
from src.experiment import (  # noqa: E402
    LOGS_SUBDIR, UNITS_SUBDIR, RunConfig, build_unseen_test_pools, setup_logging, unit_name,
)
from src.merging import get_method  # noqa: E402
from src.strategies._common import clear_cache, eval_all_tasks, unseen_fields  # noqa: E402
from src.tasks import available_tasks  # noqa: E402

SKIP_STRATEGIES = ("directional_sampling",)


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# --------------------------------------------------------------------------- #
# Cell enumeration
# --------------------------------------------------------------------------- #

def _cells(strategies: dict) -> List[Tuple[str, dict]]:
    """Flatten a unit's strategies into ("<strategy>[/<init>]", cell) pairs in a
    stable, table-friendly order."""
    order = ["baselines", "coeff_search", "weight_gd", "weight_gd_lora", "subspace_gd", "bo_search"]
    order += [s for s in strategies if s not in order]
    out = []
    for strat in order:
        if strat not in strategies or strat in SKIP_STRATEGIES:
            continue
        block = strategies[strat]
        if strat == "coeff_search":
            out.append((strat, block))
        else:
            for init, cell in block.items():
                out.append((f"{strat}/{init}", cell))
    return out


def _remap_checkpoint(path: Optional[str], recorded_root: str, new_root: str) -> Optional[str]:
    if not path:
        return None
    for prefix in (recorded_root, "checkpoints/finetuned"):
        if prefix and path.startswith(prefix.rstrip("/") + "/"):
            return os.path.join(new_root, path[len(prefix.rstrip("/")) + 1:])
    return path


def _plan_cell(path: str, cell: dict, n_tasks: int, ckpt_root_recorded: str, ckpt_root: str):
    """Return ("base"|"scalar"|"checkpoint"|"coeffs", payload) or (None, reason)."""
    strat = path.split("/")[0]
    if path == "baselines/pretrained":
        return "base", None
    if path in ("baselines/merged_avg", "baselines/merged_coeff1"):
        coeff = cell.get("coeff", 1.0 / n_tasks if path.endswith("avg") else 1.0)
        return "scalar", float(coeff)
    if strat == "coeff_search":
        return "scalar", float(cell["best_lambda"])
    ckpt = _remap_checkpoint(cell.get("checkpoint"), ckpt_root_recorded, ckpt_root)
    have_ckpt = bool(ckpt) and os.path.isdir(ckpt) and _dir_with_weights(ckpt) is not None
    if have_ckpt:
        return "checkpoint", _dir_with_weights(ckpt)
    if strat in ("subspace_gd", "bo_search") and cell.get("coefficients_final") is not None:
        return "coeffs", [float(c) for c in cell["coefficients_final"]]
    if ckpt:
        return None, f"checkpoint not found locally: {ckpt} (admin/download_finetuned.sh?)"
    return None, "no checkpoint recorded and no coefficients_final to rebuild from"


def _strip_unseen(unit: dict) -> dict:
    """Drop every unseen_* field (and the ood_eval block) from a unit copied
    as a SOURCE: the source may itself be an earlier OOD unit under another
    protocol (e.g. the zero-shot copies), and those numbers must not be
    mistaken for this run's. Seen fields are untouched."""
    for k in ("unseen_tasks", "unseen_n_shot", "ood_eval"):
        unit.pop(k, None)
    for t, st in list(unit.get("data_stats", {}).items()):
        if isinstance(st, dict) and st.get("unseen"):
            del unit["data_stats"][t]
    for strat, block in unit.get("strategies", {}).items():
        cells = [block] if strat == "coeff_search" else [c for c in block.values() if isinstance(c, dict)]
        for c in cells:
            for k in [k for k in c if k.startswith("unseen_")]:
                del c[k]
    return unit


def _needs_eval(cell: dict, unseen: List[str], force: bool, prefix: str = "unseen_") -> bool:
    if force:
        return True
    have = cell.get(f"{prefix}test_per_task") or {}
    return any(t not in have for t in unseen)


# --------------------------------------------------------------------------- #
# Weight sources (lazy: task vectors only load if some cell needs them)
# --------------------------------------------------------------------------- #

class Weights:
    def __init__(self, base_model: str, method_name: str, method_kwargs: dict,
                 tasks: List[str], arch: str, task_vectors_dir: str):
        self.base_model = base_model
        self.method = get_method(method_name, **method_kwargs)
        self.tasks, self.arch, self.task_vectors_dir = tasks, arch, task_vectors_dir
        self._base_sd = self._task_sds = self._keys = self._basis = None

    @property
    def base_sd(self):
        if self._base_sd is None:
            logging.info(f"loading base state dict: {self.base_model}")
            self._base_sd = models.load_state_dict(self.base_model, device="cpu")
        return self._base_sd

    def _load_tasks(self):
        if self._task_sds is None:
            ckpts = resolve_task_checkpoints(self.task_vectors_dir, self.tasks, self.arch)
            for t, c in zip(self.tasks, ckpts):
                logging.info(f"  task vector {t} <- {c}")
            self._task_sds = [models.load_state_dict(c, device="cpu") for c in ckpts]
            self._keys = models.shared_float_keys(self.base_sd, *self._task_sds)
        return self._task_sds, self._keys

    def scalar(self, coeff: float) -> dict:
        task_sds, keys = self._load_tasks()
        delta = self.method.merged_delta(self.base_sd, task_sds, keys, coeff=coeff)
        W = dict(self.base_sd)
        for k in keys:
            W[k] = self.base_sd[k].float() + delta[k]
        return W

    def coeffs(self, c: List[float]) -> dict:
        task_sds, keys = self._load_tasks()
        if self._basis is None:
            self._basis = self.method.basis(self.base_sd, task_sds, keys)
        if len(c) != len(self._basis):
            raise ValueError(f"{len(c)} coefficients for a {len(self._basis)}-direction basis")
        W = dict(self.base_sd)
        for k in keys:
            w = self.base_sd[k].float()
            for ci, d in zip(c, self._basis):
                w = w + ci * d[k].float()
            W[k] = w
        return W


def _load_into(model, kind: str, payload, weights: Weights) -> None:
    if kind == "base":
        model.load_state_dict(dict(weights.base_sd), strict=True)
    elif kind == "scalar":
        model.load_state_dict(weights.scalar(payload), strict=True)
    elif kind == "coeffs":
        model.load_state_dict(weights.coeffs(payload), strict=True)
    elif kind == "checkpoint":
        model.load_state_dict(models.load_state_dict(payload, device="cpu"), strict=True)
    else:
        raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unit", required=True, help="Source unit JSON (results/units/...__full.json)")
    p.add_argument("--unseen-tasks", required=True,
                   help=f"Comma-separated held-out tasks, from {available_tasks()}")
    p.add_argument("--unseen-test-samples", type=int, default=None,
                   help="Cap each unseen task's test pool (default: full)")
    p.add_argument("--n-shot", default=None,
                   help="Few-shot prompting for the unseen tasks: 'auto' (gsm8k 4 / mbpp 3), an int, or unset "
                        "for zero-shot. The output unit gains a __fewshot infix when enabled.")
    p.add_argument("--raw-prompt", action="store_true",
                   help="Build the unseen prompts as raw completions (no chat template) -- the pretrained "
                        "*-Base reference. Scores go to unseen_raw_test_* fields, so pair it with "
                        "--cells baselines/pretrained.")
    p.add_argument("--cells", default="",
                   help="Comma-separated cells to score, e.g. coeff_search,baselines/pretrained,weight_gd/avg "
                        "(default: every cell in the unit except directional_sampling)")
    p.add_argument("--force", action="store_true", help="Re-score cells that already have unseen results")
    p.add_argument("--task-vectors-dir", default="checkpoints/task_vectors")
    p.add_argument("--checkpoint-root", default="checkpoints/finetuned",
                   help="Where the unit's recorded GD checkpoints live locally (the recorded prefix is remapped)")
    p.add_argument("--out-dir", default="results/ood_generalization")
    p.add_argument("--gen-batch-size", type=int, default=None, help="Default: the unit's own")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override every unseen task's MAX_NEW_TOKENS (default: per task)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    unseen = _parse_list(args.unseen_tasks)
    if len(unseen) != len(set(unseen)):
        p.error("--unseen-tasks contains duplicates")

    src = json.load(open(args.unit))
    cfg_src = src.get("config", {})
    tasks = src["tasks"]
    overlap = set(tasks) & set(unseen)
    if overlap:
        p.error(f"unseen tasks overlap the unit's tasks: {sorted(overlap)}")

    # RunConfig only for naming + the unseen pools (same seed / split as a live run).
    cfg = RunConfig(
        base_model=src["base_model"], arch=src["arch"], method=src["method"], tasks=tasks,
        task_checkpoints=[], unseen_tasks=unseen, unseen_test_samples=args.unseen_test_samples,
        unseen_n_shot=args.n_shot, unseen_raw_prompt=args.raw_prompt,
        budget=src.get("budget", "full"), seed=int(src.get("seed", cfg_src.get("seed", 42))),
        out_dir=args.out_dir,
    )
    n_shot = {t: data.resolve_n_shot(t, args.n_shot) for t in unseen}
    prefix = "unseen_raw_" if args.raw_prompt else "unseen_"
    name = unit_name(cfg)
    out_path = Path(args.out_dir) / UNITS_SUBDIR / f"{name}.json"

    if out_path.exists():
        out = json.load(open(out_path))
        if out.get("unseen_tasks") != unseen:
            raise ValueError(f"{out_path} has unseen_tasks={out.get('unseen_tasks')}, asked for {unseen}")
        if out.get("unseen_n_shot", {t: 0 for t in unseen}) != n_shot:
            raise ValueError(f"{out_path} has unseen_n_shot={out.get('unseen_n_shot')}, asked for {n_shot}")
        # pick up cells the source unit gained since (seen numbers stay the source's)
        src = _strip_unseen(copy.deepcopy(src))
        for strat, block in src.get("strategies", {}).items():
            dst = out.setdefault("strategies", {})
            if strat == "coeff_search":
                dst.setdefault(strat, block)
            else:
                dst.setdefault(strat, {})
                for init, cell in block.items():
                    dst[strat].setdefault(init, cell)
    else:
        out = _strip_unseen(copy.deepcopy(src))
        out["unseen_tasks"] = list(unseen)
    out["unseen_n_shot"] = n_shot
    for strat in SKIP_STRATEGIES:
        out.get("strategies", {}).pop(strat, None)
    meta = out.setdefault("ood_eval", {})
    meta.update({
        "source_unit": os.path.abspath(args.unit),
        "unseen_test_samples": args.unseen_test_samples,
        "posthoc": True,
        "last_run": datetime.now().isoformat(timespec="seconds"),
    })

    # plan
    n_tasks = len(tasks)
    ckpt_root_recorded = cfg_src.get("checkpoint_root", "checkpoints/finetuned")
    wanted = set(_parse_list(args.cells)) if args.cells else None
    todo, skipped = [], {}
    for path, cell in _cells(out["strategies"]):
        if wanted is not None and path not in wanted:
            continue
        if not _needs_eval(cell, unseen, args.force, prefix):
            continue
        kind, payload = _plan_cell(path, cell, n_tasks, ckpt_root_recorded, args.checkpoint_root)
        if kind is None:
            skipped[path] = payload
            continue
        todo.append((path, cell, kind, payload))
    if wanted is not None:
        missing = wanted - {p for p, *_ in todo} - set(skipped)
        for m in missing:
            skipped.setdefault(m, "not in unit or already has every requested unseen task")
    if skipped:
        meta["skipped"] = skipped
    if not todo:
        print(f"Nothing to score for {out_path} (skipped: {skipped or 'none'})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=2, default=float)
        return

    log_path = setup_logging(args.out_dir, name)
    logging.info(f"Log -> {log_path}")
    logging.info(f"Unit: {name}  (post-hoc OOD eval of {args.unit})")
    logging.info(f"Unseen tasks: {unseen}  n_shot={n_shot}  raw_prompt={args.raw_prompt}  cells: {[t[0] for t in todo]}")
    for k, v in skipped.items():
        logging.info(f"  skip {k}: {v}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = cfg_src.get("dtype", "bfloat16")
    gen_bs = args.gen_batch_size or int(cfg_src.get("gen_batch_size", 16))
    temperature = float(cfg_src.get("temperature", 0.01))
    top_p = float(cfg_src.get("top_p", 0.95))

    tokenizer = models.load_tokenizer(src["base_model"])
    unseen_examples = build_unseen_test_pools(cfg, stats=out.setdefault("data_stats", {}))
    weights = Weights(src["base_model"], src["method"], cfg_src.get("method_kwargs", {}) or {},
                      tasks, src["arch"], args.task_vectors_dir)
    logging.info(f"loading resident model: {src['base_model']} ({dtype}) on {device}")
    model = models.load_causal_lm(src["base_model"], device=device, dtype=dtype)
    model.eval()

    def persist(reason: str) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        logging.info(f"  [saved] {reason} -> {out_path.name}")

    for path, cell, kind, payload in todo:
        desc = payload if kind != "coeffs" else [round(c, 4) for c in payload]
        logging.info(f"\n=== {path}  ({kind}: {desc}) ===")
        with torch.no_grad():
            _load_into(model, kind, payload, weights)
        clear_cache(device)
        res = eval_all_tasks(
            model, tokenizer, unseen_examples, unseen, device,
            args.max_new_tokens, temperature, top_p, gen_bs,
        )
        clear_cache(device)
        merged = dict(cell.get(f"{prefix}test_per_task") or {})
        merged.update(res)
        for k, v in unseen_fields(merged).items():
            cell[k.replace("unseen_", prefix, 1)] = v
        cell[f"{prefix}source"] = kind
        logging.info(
            f"  {path}: seen test_avg_score={cell.get('test_avg_score')}  "
            f"{prefix}test_avg_score={cell[f'{prefix}test_avg_score']:.4f}  per_task={res}"
        )
        persist(path)

    logging.info(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
