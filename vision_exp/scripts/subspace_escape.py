#!/usr/bin/env python3
"""Table + trajectories: from various points in the task-vector subspace S, does
continued multi-task GD stay in S or leave it, and how far?

Table -- one row per starting point (``--out-dir/table.csv``):
    valid_acc / valid_loss      held-out metrics of the point itself, no GD
    test_acc / test_loss        same on the test split
    grad_cos_subspace           cos(gradient, its projection onto S) at that point
    gd_final_dist_to_subspace   ||w_T - P_S(w_T)|| after a full weight_gd run from there
    gd_test_acc                 test accuracy after that run

Rows: ``pretrained``, ``--num-random`` random points on S, ``coeff_best`` (from
coeff_search), and ``merged`` (base + 1.0*tau_i, the plain task-vector sum).

Trajectories -- each row's weight_gd run is recorded in a 3D frame anchored at
the pretrained weights (e1, e2 = the orthonormalized task-vector subspace S;
z = that row's own escape direction) and stored in results.json; the figure
is drawn from it by ``scripts/plots/plot_subspace_escape.py``.

``--num-points N`` records N points along the run (evenly spaced, the last one
landing on the final step) plus the t=0 start, so N+1 per row. Every recorded
point costs a full valid AND test eval pass, which dominates runtime -- start
low and cap ``--test-samples`` while iterating.

Requires exactly 2 tasks, so S is 2-D and the 3-axis frame is well-defined.

Idempotent: everything is kept in ``--out-dir/results.json``, keyed by row name;
a rerun skips rows already there (``--force`` recomputes). ``table.csv`` is
always regenerated from it.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.checkpoints import resolve_task_checkpoints
from src.data import apply_valid_budget, build_valid_pool, get_dataloader, get_dataloader_from_dataset
from src.merging import get_method
from src.models import MultiTaskCLIPClassifier, avg_metrics, load_state_dict, shared_float_keys
from src.strategies import coeff_search, weight_gd
from src.strategies._common import eval_multitask_all
from src.strategies.context import StrategyContext
from src.subspace import SubspaceBasis, combine, vnorm, vsub

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("subspace_escape")


def probe_gradient(base_model: str, tasks: List[str], W: Dict[str, torch.Tensor], loaders: Dict[str, object],
                   device: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    """Gradient of the multi-task loss at ``W`` -- no optimizer, no step --
    averaged over one full pass of ``loaders`` (every batch of every task, each
    task weighted equally): which way continued multi-task training would
    initially pull the weights from this point."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = MultiTaskCLIPClassifier(base_model, tasks, device)
    model.model.load_state_dict({k: W[k].float().to(device) for k in keys}, strict=False)
    model.train()
    for p in model.parameters():
        p.grad = None

    for task in tasks:
        loader = loaders[task]
        n_batches = len(loader)
        for batch in loader:
            labels = batch["labels"].to(device)
            _, loss = model(batch["pixel_values"].to(device), task, labels)
            (loss / len(tasks) / n_batches).backward()

    param_by_key = dict(model.model.named_parameters())
    g = {k: param_by_key[k].grad.detach().clone() for k in keys}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return g


def parse_k_map(s: str, tasks: List[str]) -> Dict[str, str]:
    """Either a single value applied to every task, or 'task:v,task:v'.

    A value is passed through as-is to ``data.apply_valid_budget`` (via
    ``parse_budget``), which accepts an integer per-class count OR the named
    modes ``low`` / ``medium`` / ``full`` (the whole held-out valid pool, no
    subsampling) -- so ``--valid-k full`` and ``--valid-k eurosat:full,gtsrb:64``
    both work.
    """
    if ":" in s:
        out = {}
        for part in s.split(","):
            t, v = part.split(":")
            out[t.strip()] = v.strip()
        missing = [t for t in tasks if t not in out]
        if missing:
            raise ValueError(f"--valid-k missing entries for {missing}")
        return out
    return {t: s for t in tasks}


def build_loaders(tasks, valid_k, base_model, batch_size, data_seed, test_samples):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(base_model)

    def transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    valid_loaders, test_loaders, stats = {}, {}, {}
    for task in tasks:
        pool = build_valid_pool(task, seed=data_seed)
        ds, used, pool_size = apply_valid_budget(pool, valid_k[task], seed=data_seed)
        valid_loaders[task] = get_dataloader_from_dataset(
            ds, transform=transform, batch_size=batch_size, shuffle=True,
        )
        test_loaders[task] = get_dataloader(
            task, split="test", transform=transform, batch_size=batch_size,
            shuffle=False, seed=data_seed, num_samples=test_samples,
        )
        stats[task] = {"valid_k": valid_k[task], "used": used, "pool_size": pool_size}
        log.info(f"  {task}: {used}/{pool_size} valid samples ({valid_k[task]}/class)")
    return valid_loaders, test_loaders, stats


def load_results(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def assert_compatible(existing: dict, tasks: List[str], arch: str, method: str) -> None:
    checks = {"tasks": tasks, "arch": arch, "method": method}
    for k, expected in checks.items():
        if k in existing and existing[k] != expected:
            raise ValueError(
                f"--out-dir already has results.json with {k}={existing[k]!r}, this run asked for "
                f"{expected!r}. Use a different --out-dir, or --force to overwrite."
            )


def persist(path: str, tasks: List[str], arch: str, method: str, rows: Dict[str, dict]) -> None:
    with open(path, "w") as f:
        json.dump({"tasks": tasks, "arch": arch, "method": method,
                    "z_axis": {"kind": "per-row-escape"}, "rows": rows}, f)


# What a cached row must carry to be reusable, and what the table prints.
ROW_FIELDS = {"valid_acc", "valid_loss", "test_acc", "test_loss", "grad_cos_subspace",
              "gd_final_dist_to_subspace", "gd_test_acc"}
STEP_FIELDS = {"valid_acc", "valid_loss", "test_acc", "test_loss", "z_projection"}
TABLE_COLS = ["init", "valid_acc", "valid_loss", "test_acc", "test_loss",
              "grad_cos_subspace", "gd_final_dist_to_subspace", "gd_test_acc"]


def print_table(rows: List[dict]) -> None:
    cols = TABLE_COLS

    def cell(r, c):
        v = r.get(c)
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    widths = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(cell(r, c).ljust(widths[c]) for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--arch", default="vit-b-32")
    p.add_argument("--tasks", default="eurosat,gtsrb", help="Exactly 2 tasks (2-D task-vector subspace).")
    p.add_argument("--task-vectors-dir", default="checkpoints/task_vectors")
    p.add_argument("--method", default="ta")

    p.add_argument("--num-random", type=int, default=5, help="Number of random points on S to probe.")
    p.add_argument("--random-coeff-min", type=float, default=0.0)
    p.add_argument("--random-coeff-max", type=float, default=1.0)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--num-points", type=int, default=10,
                    help="How many trajectory points to record per row, evenly spaced with the last landing on "
                         "the final step. The t=0 start is recorded on top of these, so a row has N+1 points. "
                         "Each point costs a full valid + test eval pass.")

    p.add_argument("--valid-k", default="8", help="Either a single int or 'task:k,task:k'.")
    p.add_argument("--test-samples", type=int, default=None, help="Cap on the test-set eval, per task (default: no cap, full test split).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-seed", type=int, default=42)

    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-steps", type=int, default=11)


    p.add_argument("--epochs", type=int, default=3, help="weight_gd epochs, run from every row's point.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=0, help="0 = no early stop (keeps trajectories comparable).")

    p.add_argument("--out-dir", default="results/subspace_escape")
    p.add_argument("--force", action="store_true",
                    help="Ignore results.json and recompute every row (default: skip rows already there).")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) != 2:
        p.error("This experiment needs exactly 2 tasks (a 2-D task-vector subspace + 1 MTR axis = 3D figure).")
    valid_k = parse_k_map(args.valid_k, tasks)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.json")
    existing = None if args.force else load_results(results_path)
    if existing is not None:
        assert_compatible(existing, tasks, args.arch, args.method)
    rows_done: Dict[str, dict] = {} if existing is None else dict(existing.get("rows", {}))
    # The escape axis is per-row (each row's own start->end displacement), so
    # rows stay independently cacheable -- unlike a shared axis, adding a row
    # never invalidates the others.
    # Drop rows this script can no longer produce (e.g. subspace_gd_coeff, from
    # a strategy that has since been removed). Rows it CAN produce are kept even
    # when outside the current --num-random, so lowering it and raising it again
    # does not throw away expensive runs; only genuinely dead rows go.
    def _producible(name: str) -> bool:
        return name in ("pretrained", "coeff_best", "merged") or name.startswith("random_")

    # A row also has to carry everything the current table and figures need. A
    # row written before valid/test metrics were recorded per point cannot be
    # patched up -- the numbers were never measured -- so it is recomputed.
    def _current_schema(r: dict) -> bool:
        if not ROW_FIELDS <= set(r):
            return False
        traj = r.get("trajectory") or []
        return bool(traj) and STEP_FIELDS <= set(traj[-1])

    stale = [n for n in rows_done if not _producible(n)]
    outdated = [n for n in rows_done if n not in stale and not _current_schema(rows_done[n])]
    if stale:
        log.info(f"Dropping {len(stale)} stale row(s) from {results_path} "
                 f"(not producible by this script any more): {stale}")
    if outdated:
        log.info(f"Recomputing {len(outdated)} row(s) from {results_path}: written before the current "
                 f"metrics/axis existed, so they lack fields this run needs -- {outdated}")
    for n in stale + outdated:
        del rows_done[n]

    if rows_done:
        log.info(f"Loaded {len(rows_done)} already-computed row(s) from {results_path}: {sorted(rows_done)}")

    log.info(f"=== Loading checkpoints (tasks={tasks} arch={args.arch} method={args.method}) ===")
    task_ckpts = resolve_task_checkpoints(args.task_vectors_dir, tasks, args.arch)
    for t, c in zip(tasks, task_ckpts):
        log.info(f"  {t} -> {c}")
    base_sd = load_state_dict(args.base_model, device="cpu")
    task_sds = [load_state_dict(c, device="cpu") for c in task_ckpts]
    keys = shared_float_keys(base_sd, *task_sds)
    method = get_method(args.method)

    log.info("=== Building validation subset ===")
    valid_loaders, test_loaders, data_stats = build_loaders(
        tasks, valid_k, args.base_model, args.batch_size, args.data_seed, args.test_samples,
    )

    ctx = StrategyContext(
        method=method, base_model=args.base_model, base_sd=base_sd, task_sds=task_sds, keys=keys,
        tasks=tasks, valid_loaders=valid_loaders, test_loaders=test_loaders, device=device,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max, lambda_steps=args.lambda_steps,
        gd_epochs=args.epochs, gd_lr=args.lr, gd_warmup_ratio=args.warmup_ratio,
        gd_optimizer=args.optimizer, gd_momentum=args.momentum, gd_patience=args.patience,
    )

    log.info("\n=== Building S (task-vector subspace) ===")
    taus = method.basis(base_sd, task_sds, keys)
    directions_dev = [{k: t[k].float().to(device) for k in keys} for t in taus]
    subspace_basis = SubspaceBasis(directions_dev, keys)
    if subspace_basis.dim != 2:
        p.error(f"Expected a 2-D task-vector subspace, got rank {subspace_basis.dim} "
                f"(two task vectors were near-collinear).")
    base_dev = {k: base_sd[k].float().to(device) for k in keys}

    if "coeff_best" in rows_done:
        log.info("\n=== coeff_search === skipped (coeff_best row already cached)")
        best_lambda = rows_done["coeff_best"]["coeffs"][0]
    else:
        log.info("\n=== coeff_search ===")
        cs_res = coeff_search.run(ctx)
        best_lambda = cs_res["best_lambda"]
        log.info(f"  best_lambda={best_lambda:.4f}")
    ctx.best_lambda = best_lambda

    rng = random.Random(args.random_seed)
    rows_spec = [("pretrained", [0.0] * len(taus))]
    for i in range(args.num_random):
        c = [rng.uniform(args.random_coeff_min, args.random_coeff_max) for _ in taus]
        rows_spec.append((f"random_{i}", c))
    rows_spec.append(("coeff_best", [ctx.best_lambda] * len(taus)))
    # The plain task-vector sum: base + 1.0*tau_i for every task.
    rows_spec.append(("merged", [1.0] * len(taus)))

    probe_model = None  # built lazily -- skip entirely if every row is cached

    probe_model = None  # built lazily -- skip entirely if every row is cached

    def get_probe():
        nonlocal probe_model
        if probe_model is None:
            probe_model = MultiTaskCLIPClassifier(args.base_model, tasks, device)
        return probe_model

    def eval_both(model) -> dict:
        """Held-out metrics at the model's current weights, on both splits.

        Called once per recorded trajectory point, so this is what makes
        --num-points expensive: each point costs a full pass over the valid
        budget AND the test split.
        """
        was_training = model.training
        model.eval()
        v = eval_multitask_all(model, valid_loaders, tasks, device)
        t = eval_multitask_all(model, test_loaders, tasks, device)
        if was_training:
            model.train()
        v_acc, v_loss = avg_metrics(v)
        t_acc, t_loss = avg_metrics(t)
        return {"valid_loss": v_loss, "valid_acc": v_acc,
                "test_loss": t_loss, "test_acc": t_acc}

    def static_metrics(name: str, W) -> dict:
        """Everything about a starting point that needs no GD: its held-out
        metrics, and the multi-task gradient there decomposed against S."""
        m = get_probe()
        m.model.load_state_dict({k: W[k].float().to(device) for k in keys}, strict=False)
        out = eval_both(m)

        g = probe_gradient(args.base_model, tasks, W, valid_loaders, device, keys)
        g_norm = vnorm(g, keys)
        alpha_g = subspace_basis.coords(g)
        g_par_norm = sum(a * a for a in alpha_g) ** 0.5
        out["grad_cos_subspace"] = (g_par_norm / g_norm) if g_norm > 0 else float("nan")
        del g
        return out

    def finish_row(name, coeffs, stats, result, v_final, traj) -> None:
        rows_done[name] = {
            "init": name,
            "coeffs": [round(c, 4) for c in coeffs],
            "test_acc": stats["test_acc"],
            "valid_acc": stats["valid_acc"],
            "test_loss": stats["test_loss"],
            "valid_loss": stats["valid_loss"],
            "grad_cos_subspace": stats["grad_cos_subspace"],
            "gd_final_dist_to_subspace": subspace_basis.residual_norm(v_final),
            "gd_test_acc": result["test_avg_acc"],
            "trajectory": traj,
        }
        r = rows_done[name]
        log.info(
            f"  start: valid_acc={r['valid_acc']:.4f} test_acc={r['test_acc']:.4f}  "
            f"grad_cos_subspace={r['grad_cos_subspace']:.4f}  |  "
            f"after GD: dist_to_S={r['gd_final_dist_to_subspace']:.4f} test_acc={r['gd_test_acc']:.4f}"
        )

    # Which optimizer steps to record. The user asks for a POINT COUNT, not a
    # stride: spread them evenly and land the last one exactly on the final
    # step, so the trajectory always ends where training ended.
    steps_per_epoch = max(len(l) for l in valid_loaders.values())
    total_steps = steps_per_epoch * args.epochs
    n_pts = max(1, min(args.num_points, total_steps))
    record_at = sorted({-(-((i + 1) * total_steps) // n_pts) - 1 for i in range(n_pts)})
    log.info(f"\n=== Trajectory sampling === {total_steps} optimizer steps, {args.num_points} requested "
             f"-> recording at steps {record_at if len(record_at) <= 12 else str(record_at[:5])[:-1] + ', ...]'}")
    log.info(f"  plus the t=0 starting point -> {len(record_at) + 1} points per row")
    log.info(f"  each point costs a full valid + test eval pass")

    for name, coeffs in rows_spec:
        if name in rows_done:
            log.info(f"\n--- {name}: already in {results_path}, skipping (--force to redo) ---")
            continue
        log.info(f"\n--- {name}  coeffs={[round(c, 4) for c in coeffs]} ---")
        W = combine(base_dev, directions_dev, coeffs, keys)
        stats = static_metrics(name, W)

        v_init = vsub(W, base_dev, keys)
        traj = [{
            "step": -1, "epoch": -1,
            "loss": None,  # no training minibatch has been drawn yet
            "lambda_hat": subspace_basis.coords(v_init),
            "residual_norm": subspace_basis.residual_norm(v_init),
            "valid_loss": stats["valid_loss"], "valid_acc": stats["valid_acc"],
            "test_loss": stats["test_loss"], "test_acc": stats["test_acc"],
        }]

        # The escape axis is this row's own start->end displacement with S
        # removed, so it is unknown until the row finishes. Keep each recorded
        # step's S-orthogonal residual (a handful of vectors, RAM is fine at
        # this point count) and project them once the axis exists.
        residuals = {}

        def snap(step, w_now, _res=residuals):
            r = subspace_basis.residual_vector(vsub(w_now, base_dev, keys))
            _res[step] = {k: r[k].detach().to(torch.float32).cpu() for k in keys}

        torch.manual_seed(args.random_seed)
        result = weight_gd.run_one(
            ctx, name, trajectory=traj, init_weights=W, record_at=set(record_at),
            return_state=True, metrics_fn=eval_both, snapshot_fn=snap,
        )
        final_sd = result.pop("state_dict")
        v_final = vsub({k: final_sd[k].float().to(device) for k in keys}, base_dev, keys)
        del final_sd

        # z axis: (end - start) displacement, minus its component inside S.
        d_perp = subspace_basis.residual_vector(vsub(v_final, v_init, keys))
        d_perp_norm = vnorm(d_perp, keys)
        if d_perp_norm > 0:
            z_hat = {k: (d_perp[k] / d_perp_norm).detach().cpu() for k in keys}
            traj[0]["z_projection"] = 0.0  # the start is the axis origin by construction
            for pt in traj[1:]:
                r = residuals[pt["step"]]
                pt["z_projection"] = float(sum(torch.sum(r[k] * z_hat[k]) for k in keys))
        else:
            for pt in traj:
                pt["z_projection"] = 0.0
        log.info(f"  escape axis ||d_perp|| = {d_perp_norm:.4f}")
        residuals.clear()
        del d_perp, W

        finish_row(name, coeffs, stats, result, v_final, traj)
        persist(results_path, tasks, args.arch, args.method, rows_done)

    table = [rows_done[name] for name, _ in rows_spec]

    print("\n")
    print_table(table)

    # table.csv is cheap and always fully regenerated from whatever is now in
    # results.json (the source of truth); only the table rows are written here,
    # not the per-step trajectories.
    table_cols = [k for k in table[0].keys() if k != "trajectory"]
    # every row now comes from the same code path, so the key set is uniform;
    # DictWriter would raise on any row carrying an unexpected extra key.
    csv_path = os.path.join(args.out_dir, "table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table_cols)
        w.writeheader()
        w.writerows({k: r[k] for k in table_cols} for r in table)
    log.info(f"\nSaved table -> {csv_path}")


if __name__ == "__main__":
    main()
