# design.md — codebase map

A short tour of how `vision_exp` is put together. See `README.md` for how to run
the experiments.

## Design in one line

Everything is an **atomic experiment unit** = `(task_set, arch, merging_method,
budget[, seed])`, stored as one JSON of `{strategy: result}`. Non-full budgets
include a selection seed in the filename; `full` does not (shared across main
results). The valid pool partition is fixed (`VALID_POOL_SEED=42`); `seed` only
changes `apply_valid_budget` subset draws.

```
runs/full_comparison/*       ┐
runs/data_scaling/*          ├─> scripts/merge_eval.py ─> src/experiment.run_experiment
runs/s_randopt/*             ┘
                                                      │
                          builds loaders (src/data) ──┤
                          loads checkpoints (src/models)
                          picks method (src/merging)
                          runs strategies (src/strategies)
                          writes results/main_exp/units/<unit>.json

runs/cost_estimation/*  ─> scripts/profile_cost.py ─> src/cost_estimation
                          times one action / strategy (src/strategies.profile_one)
                          scales by n_actions (src/cost)
                          writes results/cost_estimation/units/<unit>.json
```

Main runs write under `results/main_exp/` (`--out-dir`, default), with
`units/`, `logs/`, `plots/`. Names:
`<n>_tasks_<tasks>__<arch>__<method>__<budget>[__seedN]`.

## Layers

### `src/` — library (imported, never run directly)

- `data.py` — dataset loading and the **fixed-seed** train/valid split
  (`_train_valid_split`, `build_valid_pool`). Copied unchanged from the original
  codebase because the trained checkpoints depend on this exact split.
  `apply_valid_budget` / `parse_budget` add integer **per-class budgets**
  (`<k>` or `<k>_per_class`) on top of the original `low`/`medium`/`full`.
- `classnames.py`, `templates.py` — zero-shot class names and prompt templates
  (copied unchanged).
- `class_subset.py` — draws the fixed class subset for `runs/smaller_9task/`
  (`--num-classes` total, split evenly over tasks, `--class-seed`); the valid
  pool, test split and zero-shot heads are all restricted to it and labels
  remapped to 0..k-1 (`data.restrict_classes`). Unit names gain `__cseed<N>`.
- `models.py` — checkpoint IO (`load_state_dict`, `save_state_dict`), zero-shot
  head construction (`extract_text_embeddings`, `ZeroShotCLIPModel`), the shared
  `MultiTaskCLIPClassifier`, and task-vector helpers (`shared_float_keys`,
  `task_vector`, `sum_task_vectors`, `avg_metrics`).
- `merging/` — pluggable merging methods (see below).
- `strategies/` — coefficient-selection strategies (see below).
- `experiment.py` — the unit runner: `RunConfig` + `run_experiment`, deterministic
  JSON naming, timestamped logging, merge-into / skip-present / `--force`,
  optional checkpoint saving.
- `cost.py` / `cost_estimation.py` — profile one warmed-up action per strategy
  (valid eval or train epoch), record peak GPU memory / wall time / vision
  image counts, convert to model FLOPs (`FlopCounterMode` on
  `get_image_features`, backward = 2× forward), and scale by `n_actions`
  (coeff_search = `lambda_steps`, weight_gd = 10, subspace_gd = 20).

- `checkpoints.py` — resolves a `(task, arch)` to its checkpoint dir inside a
  single `task_vectors/` folder (see below).

### `scripts/` — thin CLIs

- `train_single.py` — single-task fine-tuning (produces task-vector checkpoints).
- `merge_eval.py` — parses args into a `RunConfig`, resolves checkpoints, and calls
  `run_experiment`.
- `plots/plot.py` — reads `results/main_exp/units/*.json` and draws the two figures into
  `results/plots/<mode>/`: `data_scaling` (accuracy vs valid budget, one curve per
  strategy/init) and `full_comparison` (grouped bars of strategy/init across
  merging methods). Both subcommands glob the units dir, so a plot always reflects
  whatever units exist. Missing strategy/init cells are skipped, not errored.
- `profile_cost.py` — same unit key as `merge_eval.py`, writes cost-only JSON
  under `--out-dir` (default `results/cost_estimation`). λ* for
  `subspace_gd`/`coeff_best` is read from `--coeff-source-dir` (`main_exp`).
- `plots/plot_cost.py` — accuracy vs cost grid (rows = memory / time / FLOPs, columns
  = models). y is `test_avg_acc` from `--acc-dir` (`main_exp`), not the profile
  run. `weight_gd` init is `avg` for TA/DARE and `merged` for TIES/TSVM.

### `runs/` — runnable bash

- `_lib.sh` — shared helpers (`arch_base_model`, `run_merge`, `run_default_plot`);
  passes `TASKS`, `arch`, and `--task-vectors-dir` to `merge_eval.py`, then calls
  `scripts/plots/plot.py` for the matching plot (chosen from the run script's folder;
  `PLOT=0` skips).
- `train/` — the two training scripts.
- `full_comparison/` — one script per `(arch, method)`, `--budget full`.
- `data_scaling/` — one script per arch (TA), sweeping `BUDGETS`.
- `smaller_9task/` — task diversity vs sample size: ViT-B/32 TA on all 9 tasks
  but only 53 classes in total (`NUM_CLASSES`, `CLASS_SEED` 0/1/2), swept over
  budgets 1..64 per class plus full; one results root per class seed
  (`results/smaller_9task_53/cseed<N>/`).
- `init_scatters/` — `weight_gd` from true-random vision weights (`random_<seed>`),
  written to `results/weight_gd_random/`; `plot.sh` draws init-vs-GD scatters
  (`scripts/plots/plot_init_scatter.py`).
- `cost_estimation/` — `ta.sh` profiles B/32, B/16, L/14 sequentially on one
  GPU; `plot.sh` draws the cost figure (`ROWS=memory,time,flops`).
- `s_randopt/` — `directional_sampling` ablation: random samples around a merge init along
  the task-vector basis plus `g_⊥`. Writes to `results/directional_sampling/`.
  `PLOT=0` (no plot mode). `coeff_best` reads `λ*` from `results/main_exp`.
- Each script exposes an editable `TASKS=(...)`; `poc_2task_*` are 2-task smoke tests.

## Merging methods — `src/merging/`

`MergeMethod` (in `base.py`) has three hooks:

- `basis(base_sd, task_sds, keys) -> list[dict]` — the directions **subspace GD**
  optimizes (one coefficient each). TA returns the `n` per-task task vectors;
  TIES returns a single merged direction.
- `merged_delta(base_sd, task_sds, keys, coeff) -> dict` — the weight-space delta
  (added to base) at a scalar coefficient; used by `coeff_search` and to build
  `weight_gd` init points.
- `default_coeffs(n_dirs, init, best_lambda) -> Tensor` — the initial coefficient
  vector for subspace GD given an init in `{pretrained, merged, coeff_best}`.

Implementations: `task_arithmetic.py` (`ta`), `ties.py` (`ties`, with a
configurable trim `density`). Register new methods in `merging/__init__.py`
(`get_method`) — nothing downstream changes.

## Strategies — `src/strategies/`

All consume a `StrategyContext` (`context.py`) and return JSON-serializable dicts.

- `coeff_search.py` — diagonal lambda sweep on valid; reports test at lambda*.
  Returns a single dict (includes `best_lambda`, `sweep`).
- `weight_gd.py` — full-weight GD from an init point; returns `{init: result}`.
  `ctx.gd_label_source` picks the training signal: `gt` (labels), or
  `expert_soft` / `expert_hard` (unlabeled; pseudo-labels from the task experts,
  see `pseudo_labels.py`). Non-gt results are keyed `weight_gd_<source>`.
- `pseudo_labels.py` — forwards each task's unlabeled data through its expert
  `theta_t` once, caches the logits on `ctx`, and wraps the loaders so batches
  carry `expert_logits`. `step_loss` implements the three label sources;
  `soft_divergence` (js / kl) is shared with `divmerge`.
- `test_time_adaptation.py` — AdaMerging (entropy) and DivMerge (divergence to
  expert logits) as `taskwise` / `layerwise` coefficient learning over
  `method.basis`, trained on `ctx.unlabeled_loaders` (built in
  `experiment.build_unlabeled_loaders`; default = test split slice). Results
  under `strategies.adamerging.<variant>` / `strategies.divmerge.<variant>`.
- `subspace_gd.py` — **simple** autograd on the coefficient vector `c` where
  `W(c) = base + sum_i c_i * dir_i`, built via `torch.func.functional_call`. The
  coefficient count is `len(method.basis(...))`, so it works for any method (TA:
  one per task; TIES: a single scalar). No task-vector prestacking, no geometry.
  Returns `{init: result}`.
- `directional_sampling.py` — random samples around an init
  (`pretrained` / `avg` / `merged` / `coeff_best`):
  `W = W* + sum_i (α_i - λ0) dir_i + β (-g_⊥)`. `g_⊥` is the valid-loss
  gradient at `W*` minus its projection onto `method.basis`. Select on valid,
  one test eval; all draws stored under `samples`. `density_valid` is the
  fraction of draws that beat `coeff_search.valid_avg_acc` (loaded from
  `results/main_exp`, same unit), not the box center. `coeff_best` also loads
  `λ*` from that unit. Returns `{init: result}`. Writes to a side results
  root, not `main_exp`.
- `baselines.py` — four static, budget-independent references on the test set,
  no GD: `pretrained` (base model), `merged_avg` (base + `method.merged_delta(coeff=1/N)`,
  where `N = len(tasks)`), `merged_coeff1` (base + `method.merged_delta(coeff=1)`),
  and `upper_bound` (joint multi-task checkpoint, when available). The runner
  gates this on `budget=full` so cells are computed once per `(tasks, arch, method)`;
  plots pull these values from the full-budget sibling for every other budget.
  Cell computation is **incremental**: rerunning after a previous partial run
  fills in only the missing cells (e.g. `upper_bound` when the MT checkpoint
  finally appears), without recomputing the others.

`coeff_search` / `weight_gd` / `subspace_gd` each expose `profile_one` (warmup +
one timed action) used by `src/cost_estimation.py`.

`coeff_best` init points depend on `coeff_search` (its `best_lambda` is read from
the same JSON or computed first in the same run). The `avg` init corresponds to
coefficient `1/N` and is defined via `MergeMethod.default_coeffs` for subspace GD
and via `merged_delta(coeff=1/N)` for weight GD.

## Checkpoints — `checkpoints/`

- `task_vectors/` — **one** folder for all inputs: single-task dirs and
  `multitask_*` dirs side by side (matching the GCS bucket). Populate it with
  `admin/download_task_vectors.sh` and/or the training scripts.
- `finetuned/` — outputs of `weight_gd` / `subspace_gd` when `--save-checkpoints`
  is set, under `finetuned/<method>/<arch>/<tasks>/<budget>/<strategy>_<init>/`.

`src/checkpoints.py` resolves a `(task, arch)` to the actual weight directory:

1. exact `task_vectors/<task>_<arch>/` (this repo's training convention), else
2. glob `task_vectors/<task>_*<patch>*/` where `patch` comes from the arch
   (`vit-b-32 -> patch32`, `vit-b-16 -> patch16`, `vit-l-14 -> patch14`),

then descends into the highest-step `checkpoint-*` subdir that contains
`model.safetensors` (or the dir itself if the weights sit at its root). `multitask_*`
dirs are ignored for single-task resolution, and ambiguous matches raise rather
than guessing. `--task-checkpoints` bypasses resolution with explicit paths.

`resolve_multitask_checkpoint(root, tasks, arch)` finds the joint multi-task
checkpoint for the `baselines` upper bound: it globs
`multitask_<t1>+<t2>+...+<tN>_*<patch>*/` (order-sensitive, matching `cfg.tasks`)
and returns the weight-bearing subdir, or `None` when no match exists.
`--multitask-checkpoint` overrides this lookup with an explicit path.

## Result JSON shape

```jsonc
{
  "tasks": [...], "arch": "vit-b-32", "method": "ta", "budget": "full", "seed": 42,
  "base_model": "openai/clip-vit-base-patch32",
  "data_stats": { ... per-task valid/test sizes ... },
  "strategies": {
    "coeff_search": { "best_lambda": ..., "sweep": [...], "test_avg_acc": ... },
    "weight_gd":    { "pretrained": {...}, "merged": {...}, "coeff_best": {...} },
    "subspace_gd":  { "pretrained": {...}, "merged": {...}, "coeff_best": {...} },
    // directional_sampling: ablation units under results/directional_sampling/,
    // keyed by init; includes samples[] (valid-only) plus one test on the winner.
    "directional_sampling": { "coeff_best": {...} },
    // baselines: only present when this unit ran at budget=full (see runner).
    // Cells fill incrementally across reruns; `upper_bound` only present when a
    // multitask checkpoint has been resolved for this (tasks, arch).
    "baselines":    {
      "pretrained":    { "test_avg_acc": ..., "test_per_task": {...} },
      "merged_avg":    { "coeff": 0.1111, "test_avg_acc": ..., "test_per_task": {...} },
      "merged_coeff1": { "coeff": 1.0,   "test_avg_acc": ..., "test_per_task": {...} },
      "upper_bound":   { "checkpoint": "...", "test_avg_acc": ..., "test_per_task": {...} }
    }
  },
  "config": { ... }
}
```

Cost-estimation units (same filename, under `results/cost_estimation/`) store
per-strategy `{action, n_actions, action_wall_s, n_fwd_images, n_bwd_images,
action_model_flops, est_wall_s, est_model_flops, peak_mem_*_bytes}` plus
`fwd_flops_per_image`. No test metrics.

## Adding things later

- **New merging method** (WD): add a `MergeMethod` subclass + registry entry.
  If it exposes a multi-direction `basis`, subspace GD handles it
  automatically. DARE (`dare`) ships as an example: it drops-and-rescales each
  task vector before summing, and reuses TA's per-task `basis`/`merged_delta`
  shape. TSV-M (`tsvm`) ships as an example of a single-direction method that
  operates per weight *matrix* rather than per flattened parameter: it SVDs
  each task's delta, truncates to a per-task rank, and re-orthogonalizes the
  concatenated singular vectors across tasks before reconstructing one merged
  direction (same `basis` shape as TIES).
- **New strategy**: add a module under `strategies/` with a `run(ctx, ...)` and
  wire it into `experiment.run_experiment`. Add a `Series` entry in
  `scripts/plots/plot.py:build_series` if you want it drawn.
- **Upper-bound / new plots**: extend `scripts/plots/plot.py` (a new subcommand or an
  extra reference line) over `results/main_exp/units/*.json` without touching the runner.
  The three test-only references already ship as `strategies.baselines`
  (`pretrained`, `merged_coeff1`, `upper_bound`) and are drawn as horizontal
  axhlines pulled from the full-budget sibling unit.

## Experiment notes

Protocol details for the run scripts in `runs/`.

### Directional sampling (ablation)

Random samples around a merge init (`DS_INIT`, default `coeff_best`):

```
W = W* + sum_i (α_i - λ0) dir_i + β (-g_⊥)
```

`α_i ~ Unif[λ0 ± DS_ALPHA]`, `β ~ Unif[0, DS_BETA]`. `g_⊥` is one valid-set
loss gradient at `W*` with the task-vector basis projected out. The valid-best
sample is tested once; every draw is stored for later histograms.

`coeff_best` reads `λ*` from `results/main_exp/units/<same unit>.json` (not from
the ablation `OUT_DIR`). Beat density (`density_valid`) is the fraction of
samples whose valid acc exceeds that unit's `coeff_search.valid_avg_acc`, not
the sampling-box center — so every init needs that file. Results go to
`results/directional_sampling/` — they are not merged into `main_exp`.

```bash
bash runs/s_randopt/2task_vit-b-32_ta.sh
bash runs/s_randopt/vit-b-32_ta.sh
# ...vit-b-32_{ties,dare}, vit-b-16_{ta,ties,dare}
```

`PLOT=0` in these scripts skips `run_default_plot` (`plot.py` has no
`directional_sampling` mode). Set `DS_INIT`, `DS_ALPHA`, `DS_BETA`, `DS_SAMPLES`
at the top of each script.

### Cost estimation

Peak GPU memory, wall-clock, and model FLOPs — one warmed-up **action** per
strategy, times a fixed action count (not a full paper rerun):

| strategy | one action | × n_actions |
|---|---|---|
| `coeff_search` | one valid eval | `lambda_steps` (11) |
| `weight_gd` | one train epoch | 10 |
| `subspace_gd` | one train epoch | 20 |

Inits match the paper curves: `weight_gd` is `avg` for TA/DARE and `merged` for
TIES/TSVM; `subspace_gd` is `coeff_best` (λ* from `results/main_exp`). Test
accuracy is **not** re-measured — plots read `test_avg_acc` from `main_exp`.

One script runs ViT-B/32 (bs=32), B/16 (bs=8), L/14 (bs=4) in order on the
same GPU. Code accepts any method; only the TA run script is checked in.

```bash
bash runs/cost_estimation/ta.sh            # -> results/cost_estimation/units/
ROWS=memory,time,flops bash runs/cost_estimation/plot.sh
```

`PLOT=1` on `ta.sh` draws the figure at the end. `ROWS` is a subset of
`memory,time,flops` (memory x is linear; time and FLOPs are log). The figure is
three rows × one column per model, one marker per strategy.

### Choosing tasks / methods / strategies

Nothing is hardcoded. To change what runs, edit the variables at the top of the
script: `METHOD` (`ta`/`ties`/`dare`/`tsvm`), `STRATEGIES`, `WEIGHT_GD_INITS`,
`SUBSPACE_GD_INITS`, `BUDGETS`, and `TASKS`.

#### Test-time adaptation (`runs/test_time_adaptation/`)

Compares methods that adapt on **unlabeled** data. All of them train on the same
data, `--unlabeled-source test` (a slice of each task's test split, transductive
as in the AdaMerging paper; `valid` reuses the budget loaders) capped at
`--unlabeled-samples N` per task (default: whole split), and are evaluated on the
test split:

| strategy | what is learned | loss | result key |
|---|---|---|---|
| `adamerging` | coefficients (`taskwise` / `layerwise`, clamped to [0,1]) | softmax entropy | `strategies.adamerging.<variant>` |
| `divmerge` | coefficients (same variants) | `--divergence js\|kl` to task-expert logits | `strategies.divmerge.<variant>` |
| `weight_gd` + `--gd-label-source expert_soft` | full weights | same divergence to expert logits | `strategies.weight_gd_expert_soft.<init>` |
| `weight_gd` + `--gd-label-source expert_hard` | full weights | CE on expert argmax | `strategies.weight_gd_expert_hard.<init>` |

Expert logits are computed once per task (`src/strategies/pseudo_labels.py`) and
shared by `divmerge` and the `expert_*` runs; each cell records
`expert_on_unlabeled` (pseudo-label accuracy) and `unlabeled_data`. One step of
`adamerging` / `divmerge` = one batch per task (`--ada-steps`, `--ada-lr`,
`--ada-prior`; `--div-steps`, `--div-lr`, `--div-prior`; both priors default to
1/N, i.e. the `avg` init, pass `--ada-prior 0.3` for the paper's AdaMerging init). A unit refuses to merge cells trained on
different unlabeled data, so the run script puts each `UNLABELED_SAMPLES` in its
own `results/test_time_adaptation[_nN]/`. Env knobs: `UNLABELED_SOURCE`,
`UNLABELED_SAMPLES`, `DIVERGENCE`, `ADA_*`, `DIV_*`, `GD_LABEL_SOURCE`.
Ready-made 2-task smoke tests:

```bash
bash runs/full_comparison/poc_2task_ta.sh
bash runs/data_scaling/poc_2task_ta.sh
```

### Calling the runner directly

Checkpoints are resolved from `--task-vectors-dir` per `(task, arch)`:

```bash
python scripts/merge_eval.py \
  --base-model openai/clip-vit-base-patch32 --arch vit-b-32 --method ta \
  --tasks eurosat,gtsrb \
  --task-vectors-dir checkpoints/task_vectors \
  --budget full \
  --strategies coeff_search,weight_gd,subspace_gd,baselines \
  --weight-gd-inits pretrained,avg,merged,coeff_best \
  --subspace-gd-inits pretrained,avg,merged,coeff_best
```

Useful flags:

- `--task-vectors-dir` — folder with all checkpoints (default `checkpoints/task_vectors`).
  Each `(task, arch)` resolves to `<task>_<arch>/` (this repo's training) or
  `<task>_*<patch>*/` (GCS/HF naming), descending into the weight-bearing
  `checkpoint-*` subdir. Ambiguous matches raise instead of guessing.
- `--task-checkpoints` — explicit comma list (aligned with `--tasks`) that
  overrides resolution when you want exact paths.
- `--multitask-checkpoint` — explicit multi-task checkpoint used for the
  `baselines` upper bound. Defaults to auto-resolve inside `--task-vectors-dir`
  by globbing `multitask_<t1>+...+<tN>_*<patch>*/`; if nothing matches, the
  `upper_bound` cell is silently omitted.
- `--budget` — `full | medium | low | <k> | <k>_per_class`.
- `--strategies` — subset of strategies to run.
- `--weight-gd-inits` / `--subspace-gd-inits` — independent init-point lists for
  `weight_gd` and `subspace_gd` (each a subset of `pretrained,avg,merged,coeff_best`;
  `avg` = coefficient `1/N`, the task-vector average for TA).
- `--ds-inits` / `--ds-alpha` / `--ds-beta` / `--ds-samples` — `directional_sampling`
  center (`pretrained,avg,merged,coeff_best`), box, and draw count.
- `--force` — comma list of strategies to recompute even if already in the JSON.
- `--save-checkpoints` — save the `weight_gd` / `subspace_gd` finetuned weights
  (named by init point) under `checkpoints/finetuned/`.

Results are written under `results/main_exp/` (`units/`, `logs/`, `plots/`),
merging into any existing unit file (present cells are skipped unless `--force`d).
`--out-dir` overrides the results root. W&B logs to project `model_merging_main_exp`
(online by default; `WANDB=0` disables, `WANDB_MODE=offline` for local-only).
