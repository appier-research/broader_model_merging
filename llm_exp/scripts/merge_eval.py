#!/usr/bin/env python3
"""Run one merging experiment unit = (tasks, arch, method, budget).

Evaluates the requested coefficient-selection strategies (coeff_search,
weight_gd, subspace_gd, directional_sampling) and merges the results into a
deterministic JSON so
overlapping cells across runs are computed once. See the README for
end-to-end usage.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checkpoints import resolve_task_checkpoints
from src.experiment import RunConfig, run_experiment
from src import strategies
from src.merging import available_methods
from src.tasks import available_tasks


def _parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _opt_float(s: str):
    """Float CLI value, or None for 'none' (used by --bo-lower to lift the clamp)."""
    return None if s.strip().lower() in ("none", "") else float(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base", help="HF id of the pretrained base model")
    p.add_argument("--arch", default="qwen3-0.6b", help="Short arch tag used in checkpoint dir names")
    p.add_argument("--method", required=True, help=f"Merging method: one of {available_methods()}")
    p.add_argument("--tasks", required=True,
                   help=f"Comma-separated task labels (merged / selected on), from {available_tasks()}")
    p.add_argument("--unseen-tasks", default="",
                   help="Comma-separated held-out tasks (e.g. gsm8k,mbpp): test-only, no task vectors, no "
                        "valid data; every strategy's final weights get unseen_test_* fields")
    p.add_argument("--unseen-test-samples", type=int, default=None,
                   help="Cap each unseen task's test pool (default: --test-samples, else full)")
    p.add_argument("--n-shot", default=None,
                   help="Few-shot prompting for the unseen tasks: 'auto' (gsm8k 4 / mbpp 3), an int, or "
                        "unset for zero-shot. Unit name gains __fewshot when enabled.")
    p.add_argument("--task-vectors-dir", default="checkpoints/task_vectors",
                   help="Folder holding all task-vector checkpoints; resolved per (task, arch)")
    p.add_argument("--task-checkpoints", default=None,
                   help="Explicit comma-separated checkpoint paths/HF ids (aligned with --tasks); "
                        "overrides --task-vectors-dir")
    p.add_argument("--budget", default="full", help="Valid budget: full | medium | low | <k> | <k>_per_class")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--strategies", default=",".join(strategies.ALL_STRATEGIES),
                   help=f"Comma-separated strategies from {list(strategies.ALL_STRATEGIES)}")
    p.add_argument("--weight-gd-inits", default=",".join(strategies.DEFAULT_INITS),
                   help=f"Init points for weight_gd from {list(strategies.DEFAULT_INITS)} "
                        f"(also random_<seed> for true-random weights)")
    p.add_argument("--weight-gd-lora-inits", default=",".join(strategies.DEFAULT_INITS),
                   help=f"Init points for weight_gd_lora from {list(strategies.DEFAULT_INITS)}")
    p.add_argument("--subspace-gd-inits", default=",".join(strategies.DEFAULT_INITS),
                   help=f"Init points for subspace_gd from {list(strategies.DEFAULT_INITS)}")
    p.add_argument("--ds-inits", default="coeff_best",
                   help=f"Init points for directional_sampling from {list(strategies.DS_INITS)}. "
                        "'subspace_best' centers the box on subspace_gd's trained per-direction "
                        "coefficients (its best cell by valid score) instead of one scalar; it needs "
                        "subspace_gd results in this unit or in --coeff-source-dir. 'bo_best' does the "
                        "same with bo_search's best-trial coefficients.")
    p.add_argument("--bo-inits", default="coeff_best",
                   help=f"Box centers for bo_search from {list(strategies.BO_INITS)} (default: coeff_best only)")
    p.add_argument("--force", default="", help="Comma-separated strategies to recompute even if present")

    p.add_argument("--test-samples", type=int, default=None, help="Cap test set per task (shuffled); None = full")

    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override generation length for every task (default: None -- each task "
                        "uses its own MAX_NEW_TOKENS from src/tasks/*.py)")
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--gen-batch-size", type=int, default=16)

    p.add_argument("--lambda-min", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--lambda-steps", type=int, default=10)

    p.add_argument("--gd-epochs", type=int, default=5)
    p.add_argument("--gd-lr", type=float, default=3e-5,
                   help="Learning rate for weight_gd (full fine-tuning). weight_gd_lora has its "
                        "own --lora-lr and does NOT read this.")
    p.add_argument("--gd-warmup-ratio", type=float, default=0.1)
    p.add_argument("--gd-batch-size", type=int, default=4, help="Per-GPU batch size.")
    p.add_argument("--gd-grad-accum-steps", type=int, default=1,
                   help="Accumulate this many gd-batch-size batches (per task) before "
                        "optimizer.step(). Effective batch size = gd-batch-size * this; "
                        "lower gd-batch-size + higher this to cut peak activation memory "
                        "without shrinking the effective batch size.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--loss-batch-size", type=int, default=4,
                   help="Batch size for the valid-set teacher-forced NLL loss (forward only, no "
                        "grad/optimizer state). Peak memory is the logits tensor (batch_size * "
                        "max_seq_length * vocab_size), so this stays conservative -- raise it if you "
                        "have headroom.")

    p.add_argument("--subspace-epochs", type=int, default=5)
    p.add_argument("--subspace-lr", type=float, default=1e-2)
    p.add_argument("--subspace-scheduler", default="cosine",
                   help="Any name transformers.get_scheduler() accepts, e.g. linear, cosine, "
                        "cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt.")
    p.add_argument("--subspace-warmup-ratio", type=float, default=0.1)
    p.add_argument("--subspace-batch-size", type=int, default=4, help="Per-GPU batch size.")
    p.add_argument("--subspace-grad-accum-steps", type=int, default=1,
                   help="Accumulate this many subspace-batch-size batches (per task) before "
                        "optimizer.step(). See --gd-grad-accum-steps.")
    p.add_argument("--subspace-basis-device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="Where subspace_gd keeps its (1 + n_dirs) full-model base/basis copies: auto keeps "
                        "as many on the GPU as fit under --subspace-basis-headroom-gib of free memory "
                        "(directions first) and streams the rest from pinned host memory; cuda/cpu force it.")
    p.add_argument("--subspace-basis-headroom-gib", type=float, default=6.0,
                   help="Free VRAM (GiB) 'auto' basis placement keeps for activations/logits/temporaries.")

    p.add_argument("--ds-alpha", type=float, default=0.05,
                   help="Half-width of Unif[lambda0 +/- alpha] for each basis coefficient")
    p.add_argument("--ds-beta", type=float, default=0.0,
                   help="Max coefficient on -g_perp (beta ~ Unif[0, beta]). 0 (the default) skips "
                        "the gradient pass entirely and keeps every draw inside the basis subspace.")
    p.add_argument("--ds-samples", type=int, default=64, help="Number of random (alpha, beta) draws")
    p.add_argument("--ds-seed", type=int, default=42, help="RNG seed for directional_sampling")
    p.add_argument("--ds-select-metric", choices=("loss", "score"), default="loss",
                   help="Metric each draw is ranked by: 'loss' = teacher-forced valid NLL (one forward "
                        "per draw, the cheap default); 'score' = generate()+score on the valid pool "
                        "(vision_exp's literal protocol, minutes per draw). The top-5 always get the "
                        "full generate+score treatment on valid and test either way.")
    p.add_argument("--ds-eval-test", action="store_true",
                   help="directional_sampling: also generate()+score the TEST pool for every draw and "
                        "report the test-score distribution over all draws (test_dist / density_test) "
                        "instead of re-evaluating the valid-selected top-5. ~10x the per-draw cost of "
                        "the valid pass; pair with fewer --ds-samples.")
    p.add_argument("--ds-basis-device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="directional_sampling: keep W*, the basis and the gradient directions on the GPU "
                        "(fast per-draw axpy) or on the CPU (each draw combined there and copied in; "
                        "needed when 1 + n_dirs + n_grads extra model copies do not fit, e.g. per-task "
                        "gradients on qwen3-1.7b). auto picks by memory.")
    p.add_argument("--ds-reeval-center", action="store_true",
                   help="directional_sampling: re-evaluate the init point W* (valid, loss, test) in this run "
                        "instead of taking its source cell's recorded scores (bo_search / subspace_gd / "
                        "coeff_search). Inits without a source cell (pretrained/avg/merged) and the 'loss' "
                        "metric always re-evaluate.")
    p.add_argument("--bo-trials", type=int, default=50,
                   help="bo_search: total trials (each = one valid generate()+score pass), startup included")
    p.add_argument("--bo-startup-trials", type=int, default=None,
                   help="bo_search: trials before the GP starts proposing (the enqueued center + uniform "
                        "draws in the box). Default None = 2 * n_dirs + 1.")
    p.add_argument("--bo-radius", type=float, default=0.7,
                   help="bo_search: half-width of the per-coefficient box around the init center")
    p.add_argument("--bo-lower", type=_opt_float, default=0.0,
                   help="bo_search: clamp every coefficient's lower bound (default 0.0; 'none' = no clamp, "
                        "allowing negative coefficients)")
    p.add_argument("--bo-seed", type=int, default=42, help="bo_search: GPSampler / startup-draw seed")
    p.add_argument("--bo-basis-device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="bo_search: keep base + basis directions on the GPU between trials (fast) or "
                        "on the CPU (W(c) built per trial and copied in; needed when 1 + n_dirs extra "
                        "model copies do not fit, e.g. qwen3-4b). auto picks by memory.")
    p.add_argument("--coeff-source-dir", default="results",
                   help="Results root to read coeff_search (best_lambda + valid metrics) from when this "
                        "unit's own JSON has none -- directional_sampling writes to a side --out-dir")

    p.add_argument("--lora-lr", type=float, default=3e-4,
                   help="Learning rate for weight_gd_lora. Separate from --gd-lr because LoRA "
                        "trains ~1.7%% of the parameters from a zero-init B and needs a step an "
                        "order of magnitude larger; held-out search on qwen3-0.6b picked 3e-4 "
                        "for LoRA vs 3e-5 for full fine-tuning.")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (weight_gd_lora)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (weight_gd_lora)")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (weight_gd_lora)")
    p.add_argument("--lora-target-modules", default=None,
                   help="Comma-separated module names for LoRA adapters (weight_gd_lora); "
                        "default: q/k/v/o_proj + gate/up/down_proj (Qwen3/Llama-family names)")

    p.add_argument("--ties-density", type=float, default=0.2, help="TIES trim density (fraction kept)")
    p.add_argument("--dare-drop-rate", type=float, default=0.5, help="DARE drop rate (fraction zeroed)")
    p.add_argument("--dare-seed", type=int, default=42, help="DARE drop-mask seed")

    p.add_argument("--svd-device", default=None,
                   help="Where TSV-M runs its SVDs: cuda | cpu (default: cuda when visible). "
                        "The merge is a one-time cost that blocks the first lambda; on qwen3-4b "
                        "it is ~81 min on CPU vs ~9 min on GPU.")
    p.add_argument("--embedding-svd-device", default=None,
                   help="Override --svd-device for the vocab-sized embed_tokens / lm_head keys "
                        "only: cpu | cuda. Unset (the default) means they follow --svd-device. "
                        "Those two need ~17.7GB of VRAM each on qwen3-4b against <1.4GB for "
                        "every other matrix while being only ~5%% of the merge, so passing cpu "
                        "costs ~4 min and lets the merge share a card with a resident model.")

    p.add_argument("--multitask-checkpoint", default=None,
                   help="Multi-task ckpt for the 'baselines' upper_bound (defaults to auto-resolve inside "
                        "--task-vectors-dir)")

    p.add_argument("--save-checkpoints", action="store_true",
                   help="Save weight_gd / subspace_gd checkpoints (named by init)")
    p.add_argument("--checkpoint-root", default="checkpoints/finetuned",
                   help="Where --save-checkpoints writes: <root>/<method>/<arch>/<tasks>[/unseen_<u>]/<budget>/<strategy>_<init>")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    tasks = _parse_list(args.tasks)
    unseen_tasks = _parse_list(args.unseen_tasks)
    if len(tasks) < 2:
        p.error("Provide at least 2 tasks")
    if len(unseen_tasks) != len(set(unseen_tasks)):
        p.error("--unseen-tasks contains duplicates")
    overlap = set(tasks) & set(unseen_tasks)
    if overlap:
        p.error(f"--unseen-tasks overlap --tasks: {sorted(overlap)}")

    if args.task_checkpoints:
        task_ckpts = _parse_list(args.task_checkpoints)
        if len(tasks) != len(task_ckpts):
            p.error(f"--tasks ({len(tasks)}) and --task-checkpoints ({len(task_ckpts)}) must align")
    else:
        try:
            task_ckpts = resolve_task_checkpoints(args.task_vectors_dir, tasks, args.arch)
        except (FileNotFoundError, ValueError) as e:
            p.error(str(e))
        for t, c in zip(tasks, task_ckpts):
            print(f"  resolved {t} -> {c}")

    method_kwargs = {}
    if args.method.lower() == "ties":
        method_kwargs["density"] = args.ties_density
    elif args.method.lower() == "dare":
        method_kwargs["drop_rate"] = args.dare_drop_rate
        method_kwargs["seed"] = args.dare_seed
    elif args.method.lower() == "tsvm":
        method_kwargs["svd_device"] = args.svd_device
        method_kwargs["embedding_svd_device"] = args.embedding_svd_device

    cfg = RunConfig(
        base_model=args.base_model, arch=args.arch, method=args.method,
        tasks=tasks, task_checkpoints=task_ckpts, budget=args.budget, seed=args.seed,
        unseen_tasks=unseen_tasks, unseen_test_samples=args.unseen_test_samples, unseen_n_shot=args.n_shot,
        strategies=_parse_list(args.strategies),
        weight_gd_inits=_parse_list(args.weight_gd_inits),
        weight_gd_lora_inits=_parse_list(args.weight_gd_lora_inits),
        subspace_gd_inits=_parse_list(args.subspace_gd_inits),
        ds_inits=_parse_list(args.ds_inits),
        bo_inits=_parse_list(args.bo_inits),
        force=set(_parse_list(args.force)),
        test_samples=args.test_samples,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max, lambda_steps=args.lambda_steps,
        gd_epochs=args.gd_epochs, gd_lr=args.gd_lr, gd_warmup_ratio=args.gd_warmup_ratio,
        gd_batch_size=args.gd_batch_size, gd_grad_accum_steps=args.gd_grad_accum_steps,
        max_seq_length=args.max_seq_length, loss_batch_size=args.loss_batch_size,
        lora_lr=args.lora_lr, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_parse_list(args.lora_target_modules) if args.lora_target_modules else None,
        subspace_epochs=args.subspace_epochs, subspace_lr=args.subspace_lr,
        subspace_scheduler=args.subspace_scheduler,
        subspace_warmup_ratio=args.subspace_warmup_ratio, subspace_batch_size=args.subspace_batch_size,
        subspace_grad_accum_steps=args.subspace_grad_accum_steps,
        subspace_basis_device=args.subspace_basis_device,
        subspace_basis_headroom_gib=args.subspace_basis_headroom_gib,
        ds_alpha=args.ds_alpha, ds_beta=args.ds_beta, ds_samples=args.ds_samples,
        ds_seed=args.ds_seed, ds_select_metric=args.ds_select_metric, ds_eval_test=args.ds_eval_test,
        ds_basis_device=args.ds_basis_device,
        ds_reeval_center=args.ds_reeval_center,
        coeff_source_dir=args.coeff_source_dir,
        bo_trials=args.bo_trials, bo_startup_trials=args.bo_startup_trials, bo_radius=args.bo_radius,
        bo_lower=args.bo_lower, bo_seed=args.bo_seed, bo_basis_device=args.bo_basis_device,
        method_kwargs=method_kwargs, save_checkpoints=args.save_checkpoints,
        checkpoint_root=args.checkpoint_root,
        out_dir=args.out_dir, device=args.device,
        task_vectors_dir=args.task_vectors_dir, multitask_checkpoint=args.multitask_checkpoint,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
