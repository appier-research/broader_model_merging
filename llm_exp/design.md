# design.md — codebase map

A short tour of how `llm_exp` is put together, mirroring `vision_exp`'s design
for LLM task-arithmetic experiments. See `README.md` for how to run things.

## Design in one line

Same atomic unit as `vision_exp`: `(task_set, arch, merging_method, budget)`,
stored as one JSON of `{strategy: result}`. Merging inputs are already
fine-tuned LLM checkpoints — this repo does not train them (see "Checkpoints"
below); it only merges and evaluates.

```
runs/full_comparison/*  ──> scripts/merge_eval.py ──> src/experiment.run_experiment
                                      │
                  builds tasks (src/tasks) ──┤
                  loads checkpoints (src/models + src/checkpoints)
                  picks method (src/merging)
                  runs strategies (src/strategies)
                  writes results/units/<unit>.json
```

Everything a run produces lives under the results root (`--out-dir`, default
`results/`), split into `units/` (per-unit JSONs), `logs/` (run logs) and `plots/`
(figures from `scripts/plots/*.py`).

**No data-scaling experiments here.** `vision_exp` has a second main result
(`data_scaling`) that sweeps per-class valid budgets (1, 2, 4, 8, ... up to
`full`) because its image datasets are large enough per class for that sweep
to be meaningful. `llm_exp` doesn't attempt the analogous sweep: bank77's
entire labeled pool (`in_domain`, 1000 examples / 77 classes) has a minimum of
4 examples in its smallest class, so per-class budgets past ~4 would silently
degenerate into "use whatever that class has" for most classes rather than a
real budget increase. Every `runs/full_comparison/*.sh` script (other than the
`poc_2task_ta.sh` smoke test) runs `BUDGETS="full"` only; the `--budget`
CLI flag and `apply_valid_budget` machinery still exist and work (useful for
cheap smoke tests), they're just not used to build a scaling curve.

**valid:test still stays at a fixed 1:9 across every task**, but for a
different reason than `vision_exp`'s scaling headroom: `weight_gd` and
`subspace_gd` *train* on the valid pool (via `sft_target`/`build_sft_examples`),
while `coeff_search` just evaluates a scalar sweep on it — so a larger valid
pool hands the GD-based strategies more tuning data than `coeff_search` needs,
biasing any strategy comparison in their favor. Keeping every task at the same
~10% ratio (bank77 100/900, ddxplus 176/1588, ifeval 60/541, usefulness_judge
25/225) keeps that comparison fair across tasks and strategies, even though it
means bank77's per-class budget ceiling is unavoidably low (see above).

## Layers

### `src/` — library (imported, never run directly)

- `tasks/` — one module per task (`bank77.py`, `ddxplus.py`, `ifeval.py`,
  `usefulness_judge.py`), each exposing the same four-function contract (see
  below). Ported from `model-merge-transfer/llm_evals`'s two-function
  (`return_datasets`/`verify_correctness`) contract, renamed and extended with
  a third function for supervised-loss targets.
- `data.py` — the LLM analogue of `vision_exp`'s fixed-seed valid/test split:
  each task's single available eval split (bank77 `in_domain`, ddxplus `test`,
  usefulness-judge `test`) is split by a fixed seed into a **valid pool**
  (used by `coeff_search`'s lambda sweep and by `weight_gd`/`subspace_gd`'s
  training loop, via `build_sft_examples`) and a **test** set (final metrics
  only) — same split shape as `_train_valid_split`, just operating over text
  rows instead of images. Every task keeps the ratio at a fixed 90/10 (see
  "valid:test still stays at a fixed 1:9" above); `EVAL_SPLIT_SPEC` supports
  a per-task `"valid_fraction"` override but nothing currently uses it.
  ifeval overrides the dispatch entirely with cross-dataset valid/test
  sourcing (see the task contract section below). `data.py` also holds the
  tokenized SFT collation (prompt + target -> `input_ids` / `attention_mask`
  / `labels`, with the prompt span masked to `-100`).
- `models.py` — checkpoint IO (`load_causal_lm`, `load_tokenizer`,
  `load_state_dict`, `save_state_dict`) built on generic
  `AutoModelForCausalLM` / `AutoTokenizer` calls (works for any HF causal LM,
  not just Qwen3, and handles sharded safetensors automatically via
  `transformers` — no manual safetensors parsing like `vision_exp` does for
  CLIP), batched `generate()` for scoring, and the same task-vector helpers as
  `vision_exp` (`shared_float_keys`, `task_vector`, `sum_task_vectors`,
  `scale_vector`, `apply_delta`, `avg_metrics`) — these are pure
  `dict[str, Tensor]` ops, ported unchanged. No per-task heads: an LLM already
  has one output layer, so "multitask" here just means the same shared decoder
  weights evaluated/trained across task-specific prompts.
- `merging/` — pluggable merging methods, ported ~verbatim from `vision_exp`
  (same `MergeMethod` interface; state-dict-only, architecture agnostic).
- `strategies/` — coefficient-selection strategies (see below), adapted for
  next-token generation / cross-entropy loss instead of image classification.
  The GD strategies' training loss is `_common.lm_loss`: the decoder as usual,
  then Liger's fused linear cross-entropy instead of `model(**batch).loss`, so
  the [seq, vocab] logits (~3.7 GiB per 2048-token sequence on Qwen's 152k
  vocab) are never materialized. Same loss and gradients (fp32 check: 1e-4).
- `cost.py` / `cost_estimation.py` — profile one warmed-up action per
  strategy (coeff_search valid generate+score, a bo_search trial, or a GD train epoch), record peak GPU
  memory / wall time / token counts, convert to model FLOPs
  (`FlopCounterMode` on one teacher-forced forward gives FLOPs per token;
  backward = 2× forward, +1× for gradient-checkpointing recompute), and scale
  by `n_actions` (coeff_search = `lambda_steps`, bo_search = `bo_trials`,
  weight_gd_lora / weight_gd = `gd_epochs`, subspace_gd = `subspace_epochs`).
  `coeff_search` / `bo_search` / `weight_gd_lora` / `weight_gd` /
  `subspace_gd` each expose a `profile_one` (warmup + one timed action) that
  shares its loop with `run_one` (`_one_epoch` / `_valid_pass` / `_valid_score`).
  The benchmark's GD strategies are `weight_gd_lora` and `subspace_gd`;
  full-parameter `weight_gd` is opt-in.
- `experiment.py` — the unit runner: `RunConfig` + `run_experiment`, same
  deterministic JSON naming, timestamped logging, merge-into / skip-present /
  `--force` as `vision_exp`.
- `checkpoints.py` — resolves a `(task, arch)` to its checkpoint dir inside
  `checkpoints/task_vectors/` (see below); simpler than `vision_exp`'s version
  since there's no CLIP-style patch-token glue (`vit-b-32 -> patch32` etc.) to
  worry about — just an exact `<task>_<arch>` match, then descend into the
  highest-step `checkpoint-*` subdir. Also resolves the optional joint
  multi-task checkpoint (`multitask_<t1>+<t2>+...+<tN>_<arch>`) for the
  `baselines` strategy's `upper_bound` cell, returning `None` when absent
  (true today) instead of raising.

### `scripts/` — thin CLIs

- `merge_eval.py` — parses args into a `RunConfig` and calls `run_experiment`
  (analogue of `merge_eval.py`). Task vectors are trained with axolotl, not a
  repo script: see `runs/train/` and `configs/train/`.
- `profile_cost.py` — same unit key as `merge_eval.py`, writes a cost-only JSON
  under `--out-dir` (default `results/cost_estimation`) via
  `src/cost_estimation.py`. λ* for the `coeff_best` init is read from
  `--coeff-source-dir` (default `results`, the main root).
- `plots/plot_cost.py` — score vs cost grid (rows = memory / time / FLOPs, columns =
  models). y is `test_avg_score` from `--acc-dir` (the main root), looked up
  at the init the cost unit records -- the best-scoring cell per strategy on
  the main results: `weight_gd_lora` `coeff_best` for TA/DARE and `merged` for
  TIES/TSVM, `subspace_gd` `merged`, `bo_search` `coeff_best` (opt-in
  `weight_gd`: `avg` for TA/DARE, `merged` for TIES/TSVM).

### `runs/` — runnable bash

- `_lib.sh` — shared helpers, same shape as `vision_exp`'s (`run_eval`, and a
  `run_default_plot` equivalent once plotting exists).
- `train/` — axolotl training of the task vectors (configs in
  `configs/train/<arch>/`).
- `full_comparison/` — one script per `(arch, method)` combo, e.g. `qwen3-0.6b_ta.sh`,
  `qwen3-0.6b_ties.sh`, plus a `*_grid.sh` per combo: the refined
  `coeff_search` grid (subspace upper bound), written to its own
  `results/<method>_coeff_grid/` root so it never touches the main units.
- `bo_search/` — one script per `(arch, method)` (qwen3-0.6b / 1.7b / 4b and
  llama3.2-1b x ta / dare / ties / tsvm) running `bo_search` alone into the main unit, which
  must already hold `coeff_search`. Knobs `BO_INITS`, `BO_TRIALS`,
  `BO_STARTUP_TRIALS`, `BO_RADIUS`, `BO_LOWER`, `BO_SEED`, `BO_BASIS_DEVICE`
  (forwarded by `_lib.sh` only when set). 1-D methods default to 20 trials,
  TA/DARE to 50. `BO_RADIUS` defaults to 0.7 (the 0.6b x TA sweep in
  `results/bo_radius/`: 0.3 pinned two coefficients at the edge, test rose
  0.678 -> 0.702 at 0.7 and was flat past it). Queue one size per GPU, e.g.
  `GPU=0 TAG=bo bash admin/run_queue.sh runs/bo_search/qwen3-0.6b_*.sh`.
- `cost_estimation/` — `ta.sh` profiles Qwen3-0.6B, 1.7B and 4B sequentially
  on one GPU under a fixed protocol (same per-step batch for every strategy, basis
  on the CPU for bo_search and on the GPU for subspace_gd, no grad accum;
  `ARCHES` overrides the list; a spec's optional 4th field pins that arch's
  strategies); `plot.sh` draws the cost figure (`ROWS=memory,time,flops`).
- `s_randopt/` — `directional_sampling` ablation: random samples around a merge init along
  the method basis plus `g_⊥`. Writes to `results/directional_sampling/`;
  `coeff_best` (and the density threshold) read `coeff_search` back out of
  `$COEFF_SOURCE_DIR`, the main results root. Set `DS_INIT`, `DS_ALPHA`,
  `DS_BETA`, `DS_SAMPLES`, `DS_SEED`, `DS_SELECT_METRIC` at the top of each
  script; `_lib.sh` forwards each only when set.

### `admin/` — operator scripts, not part of the library

- `run_queue.sh <script>...` — runs the given run scripts back-to-back on one
  GPU (`GPU` env var, default `0`), logging each to its own file under
  `admin/logs/` (`TAG` labels the log dir). Same script as
  `vision_exp/admin/run_queue.sh`.
- `download_checkpoints.sh` — `gsutil rsync` the per-task checkpoints for one
  arch (`ARCH=`, default `qwen3-0.6b`) from GCS into
  `checkpoints/task_vectors/<task>_<arch>/checkpoint-<step>/`.

## Task contract — `src/tasks/`

Every task module exposes:

- `load_split(hf_split: str) -> list[dict]` — loads and formats one HF split
  into row dicts with either `prompt` or `messages`, plus whatever `score()`
  needs (`label`, `label_text`, `data`, ...). Ported from `return_datasets()`.
- `EVAL_SPLIT_SPEC: dict` — `{"pool": "<hf_split>"}`: `src/data.py:build_valid_test`
  loads that one split and does a fixed-seed 90/10 split into valid/test (an
  optional `"valid_fraction"` key overrides the ratio, though nothing
  currently uses it — see "valid:test still stays at a fixed 1:9" above). The
  `{"valid": ..., "test": ...}` native-split-pair form is used only by the
  unseen task gsm8k (train/test): ddxplus's native `validate` split turned out
  to already be part of its checkpoint's fine-tuning data, so `tasks/ddxplus.py`
  deliberately never loads it — both valid and test come from `test` alone,
  through the ordinary "pool" path.
- `format_example(row) -> str | list[dict]` — same `prompt` / `messages`
  branching `model-merge-transfer/llm_evals/eval.py:format_prompt` uses.
- `score(response_str, row) -> bool | float` — ported from
  `verify_correctness()`. Only `ifeval` returns a float (fraction of
  instructions followed); the rest return bool.
- `sft_target(row) -> Optional[str]` — the gold completion text used as the
  supervised-loss target in `weight_gd` / `subspace_gd`, built from the same
  valid-pool rows `coeff_search` scores (`src/data.py:build_sft_examples`) —
  train and validate on the same package of data, same convention
  `vision_exp` uses (valid is a tuning budget, not a hidden holdout; test is
  the only true holdout). `bank77` / `ddxplus` return `"<label>. <label_text>"`;
  `usefulness_judge` returns `row['prediction']`.
- `ENABLE_THINKING: bool` — whether Qwen3's chat template is rendered with
  thinking left up to the model (`True`/omitted, the tokenizer's own default)
  or forced into an empty, closed `<think></think>` block (`False`), threaded
  through `src/models.py:format_prompt`/`generate` (resolved per task in
  `src/strategies/_common.py:eval_task_model` and, for the SFT training
  prompt, in `weight_gd`/`subspace_gd`'s `_build_dataloaders` via
  `data.get_sft_dataloader`'s `enable_thinking` arg) so the train-time prompt
  and the eval-time generation prompt stay consistent. All four tasks set this
  to `False`: none of their gold completions contain a `<think>` block, and
  Qwen3-0.6B's *base* model reliably opens one anyway when left to its own
  default (empirically confirmed — 150-200+ tokens of reasoning before, or
  instead of, an answer). Left uncontrolled, this breaks more than just
  speed: `ddxplus.score()` reads only the first response line, so a thinking
  preamble makes it literally unscorable (not just slower); it also makes
  weaker/base merge states pay for (and sometimes never finish) reasoning
  that stronger/fine-tuned states skip, biasing any strategy or coefficient
  comparison in favor of whichever state "thinks" less by chance. Disabling
  it everywhere keeps the comparison about merge quality, not reasoning
  behavior. `weight_gd`/`subspace_gd`'s few-epoch SFT on a task's `sft_target`
  (which never contains `<think>`) pushes the *trained* model toward skipping
  it too, but that can't be relied on for untrained states (`baselines`'
  `pretrained` cell, `coeff_search`'s weak-coefficient sweep, `merged_avg`) --
  hence forcing it at the template level rather than depending on training to
  suppress it.
- `MAX_NEW_TOKENS: int` — the generation length used for this task by every
  strategy's `eval_all_tasks` call, resolved in `src/strategies/_common.py`
  (falls back to 256 if a task module omits it), **only valid together with
  `ENABLE_THINKING=False` above** -- these numbers assume the model skips
  straight to its answer. Sized from actual generations (`Qwen/Qwen3-0.6B`,
  greedy, `ENABLE_THINKING=False`, 20-example sample per task), not just gold
  length: `bank77`/`ddxplus` (16 -- gold `"<label>. <label_text>"` tops out at
  ~12 tokens, observed generations topped out at 10-12), `usefulness_judge`
  (16, matching the other two, though it could go as low as ~8: `"YES"`/`"NO"`
  each tokenize to one token and the verdict always leads; a trailing
  justification, observed up to ~48 tokens on ~10% of responses, doesn't need
  budget because `score()` substring-matches the *whole* remaining text, not
  just the verdict word, so truncating it changes nothing but wall-clock),
  `ifeval` (256 -- free-form prose satisfying several instructions at once,
  no short canonical answer, not independently
  re-measured). Left-padded batched `generate()` runs every sequence in a
  batch to the *longest* one's stopping point, so before this
  per-task split, ifeval-sized budgets were silently paid on every task's
  generation call. `--max-new-tokens` overrides this uniformly across tasks
  when explicitly set (e.g. for a fast smoke test); its default is `None`,
  meaning "use each task's own value".

Optionally, a task module can define `build_valid_test(seed) -> (valid, test)`
to bypass `EVAL_SPLIT_SPEC` entirely — currently just `ifeval` (see below).

**ifeval is special.** Its own eval set (`google/IFEval`) has no canonical
correct completion (correctness is checked against `kwargs` instruction rules,
not a fixed string), so it can't supply both a valid pool *and* an `sft_target`
from the same source the way the other three tasks do. `tasks/ifeval.py`
overrides `build_valid_test`: **valid** (60 examples, pre-budget — a 1:9 ratio
against the 541-example test pool, same as every other task) is sampled from
`argilla/ifeval-like-data`'s `filtered` config, kept only where the response
satisfies its own instructions (`prompt_level_strict_acc`) — these rows carry
a real `response` field, so `sft_target` returns it and the generic
`build_sft_examples` path works unmodified, same as every other task. **test**
stays the full, untouched `google/IFEval` benchmark (541 examples) — never
trained or tuned on, the true holdout signal. Requires the `langdetect`
package (undeclared transitive dependency of `lighteval`'s
`language:response_language` instruction checker).

## Merging methods — `src/merging/`

Unchanged from `vision_exp` (`base.py`, `task_arithmetic.py`, `ties.py`, `dare.py`) —
`basis` / `merged_delta` / `default_coeffs` operate on plain
`dict[str, Tensor]` state dicts, so nothing about the interface is
vision-specific. Register new methods in `merging/__init__.py` (`get_method`).

`tsv_m.py` (TSV-M) is the one method that needed LLM-specific changes, both
for memory/dtype rather than for the algorithm:

- It builds each task's delta one key at a time instead of calling
  `task_vectors()`, which would materialize `n_tasks` full model-sized
  copies before the first SVD runs.
- Each matrix is upcast to fp32 for its SVD and cast back to bf16 after:
  `torch.linalg.svd` has no bf16 CPU kernel at all, and an SVD is
  numerically delicate. Only one layer is ever held in fp32.

Both `basis` and `merged_delta` go through a cached `_merged()` keyed on the
input identities (same convention as `ta`/`ties`), so `coeff_search`'s lambda
sweep pays for the SVDs once — they dominate the method's runtime.

How much they dominate is worth knowing before you run TSV-M: those SVDs are
CPU-only and took **~5 hours** before the first lambda on a 4-task unit,
roughly 8x the entire 41-point sweep that follows. The run looks stalled
(busy CPU, idle GPU) but is not. See
[`notes/tsvm_svd_cpu_bottleneck.md`](notes/tsvm_svd_cpu_bottleneck.md) for the
measurements and for why moving the SVDs to GPU would not OOM.

## Strategies — `src/strategies/`

All consume a `StrategyContext` and return JSON-serializable dicts, same shape
as `vision_exp`:

- `coeff_search.py` — diagonal lambda sweep on valid; at each lambda,
  generate + score on the valid pool, pick the best by average score, report
  test at that lambda. No gradients.
- `weight_gd.py` — full-weight GD of the shared decoder from an init point
  (`pretrained` / `avg` / `merged` / `coeff_best`), minimizing per-task
  next-token cross-entropy on `(prompt, sft_target)` pairs from the valid pool
  (ifeval substitutes its `argilla` pool — see above). Returns `{init: result}`.
- `subspace_gd.py` — autograd on the coefficient vector `c` where
  `W(c) = base + sum_i c_i * dir_i`, built via `torch.func.functional_call` on
  the causal LM, same as `vision_exp`. Coefficient count is
  `len(method.basis(...))`. Returns `{init: result}`.
- `directional_sampling.py` — random samples around an init
  (`pretrained` / `avg` / `merged` / `coeff_best` / `subspace_best`):
  `W = W* + sum_i (α_i - λ0) dir_i + β (-g_⊥)`, where `g_⊥` is one valid-loss
  gradient at `W*` (over `ctx.sft_pools`, the same pairs `weight_gd` trains on)
  minus its projection onto `method.basis`. Select on valid, then evaluate the
  valid-best five on valid *and* test; all draws are stored under `samples`.
  Returns `{init: result}`. Writes to a side results root, not `results/units/`.
- `bo_search.py` — Bayesian optimization (optuna `GPSampler`, log-EI) over the
  same coefficient vector `c` as `subspace_gd`, `W(c) = base + sum_i c_i dir_i`,
  but maximizing `valid_avg_score` (one generate()+score pass per trial)
  instead of descending the SFT loss — the loss and the score disagree
  systematically on the units run so far. Box = init center ± `bo_radius`,
  clamped at `bo_lower`; the center is the first trial, then `2*n_dirs`
  uniform startup draws, then GP proposals (`bo_trials` in total). No other
  strategy's observations are fed in, so it stands alone apart from lambda*.
  Best observed trial is reported; every trial is stored under `trials`.
  Base + basis stay on the GPU when they fit (`bo_basis_device=auto`, ~60%
  of the card), else `W(c)` is built on the CPU each trial — this is how it
  runs on qwen3-4b x TA on 24 GB, where subspace_gd's resident copies cannot.
  Returns `{init: result}` (default init: `coeff_best` only).

  Three things differ from `vision_exp`'s version, all forced by generation
  being the expensive operation here:

  1. **Per-draw metric.** vision scores every draw with the same cheap
     classification-head accuracy it uses everywhere; the analogue here is a
     full autoregressive `generate()` over every valid pool per draw — minutes
     each, so 64 draws is hours before the top-5 test evals even start.
     `ctx.ds_select_metric` defaults to `"loss"` (teacher-forced valid NLL, one
     forward per draw, lower = better) with `"score"` available for the literal
     vision protocol. `density_valid` compares against the init point's own
     valid metric on whichever axis was used (the init is evaluated up front
     with valid score, valid loss and test score, stored as `center_*`, and
     the score its source cell recorded is stored as `init_source_*`).
  2. **`ds_beta` defaults to 0**, which makes the whole `-g_⊥` axis a no-op and
     skips the backward pass (and the full-model-sized gradient it would leave
     resident) entirely. vision defaults it nonzero.
  3. **No `functional_call`.** vision substitutes `W*` into the model
     functionally; here `W*` is loaded into a real model and `dL/dW` is read off
     `.grad` after an ordinary backward — same reasoning as `subspace_gd` (see
     its docstring on gradient checkpointing). Tied weights make that lookup
     subtle: `named_parameters()` is the only view that carries `.grad` but
     drops a tied weight's duplicate name (Qwen3 ties `lm_head.weight` to
     `model.embed_tokens.weight`), while `state_dict()` keeps every name but
     returns detached tensors — so the two are joined on `data_ptr()`, and both
     aliases resolve to the one shared gradient.

  The center is a **per-direction vector** internally (`lams`), not a scalar:
  the four scalar inits broadcast one value across every direction, so their
  behavior is bit-identical to the scalar implementation, while
  `subspace_best` / `bo_best` supply one center per direction — `subspace_gd`'s
  trained / `bo_search`'s best-trial `coefficients_final`, picked from that
  strategy's best cell by `valid_avg_score` and read out of `coeff_source_dir`. For a uniform center `W*` is built via
  `method.merged_delta(coeff=λ0)` so each method keeps its own semantics (TIES
  trims, DARE drops-and-rescales — their merged delta is not a combination of
  the basis at all); a non-uniform center has no such scalar form, so `W*` is
  built from the basis directly as `sum_i c_i * dir_i`, exactly the
  reparametrization subspace_gd optimizes. `center_coeff` stays in the JSON for
  a uniform center and is `null` otherwise; `center_coeffs` always holds the
  full vector.

  Vector algebra is streamed per key (fp64 dot-product accumulators) rather than
  `torch.cat`-ed into one flat vector the way vision does: flattening a
  multi-billion-parameter state dict would cost another full-model copy per
  direction on top of the `1 + n_dirs` already resident.

- `baselines.py` — four static, budget-independent references on the test set,
  no GD: `pretrained` (base model), `merged_avg` (base +
  `method.merged_delta(coeff=1/N)`, where `N = len(tasks)`), `merged_coeff1`
  (base + `method.merged_delta(coeff=1)`), and `upper_bound` (joint multi-task
  checkpoint, when available — none exist yet, see `checkpoints.py` above).
  The runner gates this on `budget=full` so cells are computed once per
  `(tasks, arch, method)`. Cell computation is **incremental**: rerunning after
  a previous partial run fills in only the missing cells.

`coeff_best` init points depend on `coeff_search` (its `best_lambda` is read
from the same JSON or computed first in the same run). The `avg` init
corresponds to coefficient `1/N` and is defined via
`MergeMethod.default_coeffs` for `subspace_gd` and via
`merged_delta(coeff=1/N)` for `weight_gd`.

## Checkpoints — `checkpoints/`

- `task_vectors/<task>_<arch>/checkpoint-<step>/` — one directory per
  fine-tuned task checkpoint, populated by `admin/download_checkpoints.sh`
  from
  `gs://<YOUR_BUCKET>/llm_checkpoints/`.
  One set per arch (`qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `llama3.2-1b`; see
  `README.md` for the base models). Trained with axolotl from
  `configs/train/<arch>/<task>.yaml` (full fine-tuning, lr 4e-5, effective
  batch 36, 6 epochs) by `runs/train/train_all_single_task_checkpoints.sh`,
  which writes straight into this layout. The 0.6b/1.7b configs are the
  originals; the 4b/llama ones are reconstructed from the checkpoint names.
  The downloaded set pins specific steps (`admin/download_checkpoints.sh`),
  while a fresh run is resolved to its highest-step checkpoint.
- `finetuned/` — outputs of `weight_gd` / `subspace_gd` when
  `--save-checkpoints` is set, same layout convention as `vision_exp`
  (`finetuned/<method>/<arch>/<tasks>/<budget>/<strategy>_<init>/`).

`src/checkpoints.py` resolves `(task, arch)`:

1. exact `task_vectors/<task>_<arch>/`, then
2. descend into the highest-step `checkpoint-*` subdir that contains model
   weights (safetensors, sharded or not).

Ambiguous or missing matches raise rather than guessing. `--task-checkpoints`
bypasses resolution with explicit HF paths/ids.

## Model loading — `src/models.py`

Deliberately generic: `AutoModelForCausalLM.from_pretrained` /
`AutoTokenizer.from_pretrained` for both local checkpoint dirs and HF hub ids,
so any causal LM architecture works, not just Qwen3 (requires
`transformers>=4.51.0` for native `Qwen3ForCausalLM` support, no
`trust_remote_code`). Default inference path is plain `transformers.generate`
(batched, left-padded) so merged/patched state dicts can be loaded directly
-- the strategies merge in place on every step, so there is no vLLM backend.

## Result JSON shape

Same shape as `vision_exp`:

```jsonc
{
  "tasks": [...], "arch": "qwen3-0.6b", "method": "ta", "budget": "full", "seed": 42,
  "base_model": "Qwen/Qwen3-0.6B-Base",
  "data_stats": { ... per-task valid/test sizes ... },
  "strategies": {
    "coeff_search": { "best_lambda": ..., "sweep": [...], "test_avg_score": ... },
    // With --unseen-tasks (or via scripts/ood_eval.py post-hoc) every packed
    // cell also carries unseen_test_avg_score / unseen_test_per_task; the
    // seen valid_/test_avg_score never include them. Such units live under
    // results/ood_generalization/ with an __unseen_<tasks> name infix and a
    // top-level "unseen_tasks" (post-hoc ones also an "ood_eval" block).
    "weight_gd":    { "pretrained": {...}, "avg": {...}, "merged": {...}, "coeff_best": {...} },
    "subspace_gd":  { "pretrained": {...}, "avg": {...}, "merged": {...}, "coeff_best": {...} },
    // directional_sampling: ablation units under results/directional_sampling/,
    // keyed by init; includes samples[] (selection metric only) plus a full
    // generate+score on valid and test for the top-5.
    "directional_sampling": { "coeff_best": {...} },
    // baselines: only present when this unit ran at budget=full. Cells fill
    // incrementally across reruns; upper_bound only present once a multitask
    // checkpoint has been resolved for this (tasks, arch) -- none exist yet.
    "baselines":    {
      "pretrained":    { "test_avg_score": ..., "test_per_task": {...} },
      "merged_avg":    { "coeff": 0.25, "test_avg_score": ..., "test_per_task": {...} },
      "merged_coeff1": { "coeff": 1.0,  "test_avg_score": ..., "test_per_task": {...} },
      "upper_bound":   { "checkpoint": "...", "test_avg_score": ..., "test_per_task": {...} }
    }
  },
  "config": { ... }
}
```

Cost-estimation units (same filename, under `results/cost_estimation/`) store
per-strategy `{init, action, n_actions, action_wall_s, n_fwd_tokens,
n_bwd_tokens, grad_checkpointing, action_model_flops, est_wall_s,
est_model_flops, peak_mem_*_bytes}` plus a top-level `fwd_flops_per_token`.
No scores.

## Adding things later

- **New merging method**: same as `vision_exp` — add a `MergeMethod` subclass
  + registry entry.
- **New task**: add a module under `tasks/` with the four-function contract
  and register it in `tasks/__init__.py`.
- **New strategy**: add a module under `strategies/` with a `run(ctx, ...)`
  and wire it into `experiment.run_experiment`.
- **Data-scaling sweep**: intentionally out of scope for this pass (see
  "No data-scaling experiments here" above) since bank77's data volume can't
  support it meaningfully. If a future task has enough per-class data, add a
  `runs/data_scaling/` folder mirroring `vision_exp`'s and pass `--budget`
  through as usual — `apply_valid_budget` already supports it.

## Experiment notes

Protocol details for the run scripts in `runs/`.

### Directional sampling (ablation)

Random samples around a merge init (`DS_INIT`, default `coeff_best`):

```
W = W* + sum_i (α_i - λ0) dir_i + β (-g_⊥)
```

`α_i ~ Unif[λ0 ± DS_ALPHA]`, `β ~ Unif[0, DS_BETA]`. `g_⊥` is one valid-set
loss gradient at `W*` (over the same SFT pools `weight_gd` trains on) with the
method basis projected out. Every draw is stored for later histograms. What
gets a **test** score depends on `DS_EVAL_TEST`:

- `DS_EVAL_TEST=1` (the TA/TIES scripts below): **every draw** is scored on
  test as well as valid, and the cell reports the distribution of test scores
  over the whole box — `test_dist` = median / quartiles / min / max and
  `density_test`, the fraction of draws beating the init point's *own* test
  score. Nothing is selected on valid, so valid/test mismatch can neither hide
  good regions nor manufacture a lucky top-1; the `top5` rows are just read off
  the per-draw evals. A draw costs a valid pass **plus** a test pass (~10x the
  valid pass: 3254 vs 361 examples). The TA/TIES scripts use 64 draws and
  `DS_ALPHA=0.05`, and write to `results/directional_sampling_test_alpha0.05/`.
- unset (the llama script, and every run before 2026-09-14): only the
  valid-best five are re-evaluated on valid **and** test (`top5`,
  `top5_test_avg_score`). Results go to `results/directional_sampling/`.

```bash
bash runs/s_randopt/qwen3-0.6b_ta.sh
bash runs/s_randopt/qwen3-0.6b_ties.sh
bash runs/s_randopt/qwen3-1.7b_ta.sh
bash runs/s_randopt/qwen3-1.7b_ties.sh
bash runs/s_randopt/qwen3-0.6b_dare.sh   # bo_best center
bash runs/s_randopt/qwen3-1.7b_dare.sh   # bo_best center
bash runs/s_randopt/llama3.2-1b_ta.sh
```

Two knobs differ from `vision_exp`'s version, because generation is the
expensive part here:

- **`DS_SELECT_METRIC`** (default `loss`) picks what each draw is ranked by.
  `loss` is the teacher-forced valid NLL — one forward pass per draw, so 64
  draws stay cheap; `score` is `vision_exp`'s literal protocol (generate +
  score the whole valid pool per draw), which costs minutes per draw. Either
  way, the top-5 get the full generate+score treatment on valid and test, so
  their numbers are directly comparable to every other strategy.
- **`DS_BETA`** defaults to `0`, which drops the `-g_⊥` axis entirely and skips
  the backward pass (and the full-model-sized gradient it leaves resident).
  Set it `>0` to sample off the basis subspace.
- **`DS_BASIS_DEVICE`** (default `auto`) is where `W*`, the basis and the
  gradient directions sit between draws. `auto` keeps them on the GPU only if
  `1 + n_dirs + n_grads` model copies fit next to the model in ~60% of the
  card; otherwise they live
  on the CPU and each draw is combined there in fp32 and copied into the
  model key by key — seconds per draw against minutes of generate(). Host RAM
  then holds those copies on top of the fp32 base/task state dicts (~60 GB
  for 1.7b).

`DS_INIT` also accepts **`subspace_best`**, which centers the box on
`subspace_gd`'s trained coefficient vector (its best cell by valid score)
instead of a scalar. This is the one init where each direction gets its **own**
center — `α_i ~ Unif[c_i ± DS_ALPHA]` with `c` the trained vector — because a
scalar center cannot express what subspace_gd actually learned (e.g. pulling
one task's coefficient down while pushing another's up). It reads
`coefficients_final` from `$COEFF_SOURCE_DIR` the same way `coeff_best` reads
`λ*`, and errors clearly if no `subspace_gd` results are there.

**`bo_best`** is the same kind of per-direction center, read from
`bo_search`'s best cell (its best trial's `coefficients_final`) instead. Use it
where BO beat both the coefficient sweep and `subspace_gd` on valid, i.e. where
BO's point is the strongest known center to sample around.

`coeff_best` reads `λ*` from `$COEFF_SOURCE_DIR` (default `results/`, the main
results root) when this unit's own JSON has no `coeff_search`. The init point
`W*`'s valid/test scores (`center_*`) are, by default, **the ones its source
cell recorded** (`bo_search` for `bo_best`, `subspace_gd` for `subspace_best`,
`coeff_search` for `coeff_best`; also kept as `init_source_*`) — the same
numbers the main results report, so beat density (`density_valid` /
`density_test`, the fraction of draws beating **the init point's own** score)
is measured against them and not against a re-generation that differs by
sampling noise. `W*` is re-evaluated in the run only for inits with no source
cell (`pretrained` / `avg` / `merged`), for `DS_SELECT_METRIC=loss`, or with
`DS_REEVAL_CENTER=1`; `center_source` records which. `plot_ds_test_dist.py`
marks the recorded score too (`--center reeval` for the other). Results go to
`results/directional_sampling/`, not the main `results/units/`.

Set `DS_INIT`, `DS_ALPHA`, `DS_BETA`, `DS_SAMPLES`, `DS_SEED`, `DS_SELECT_METRIC`,
`DS_EVAL_TEST` and `DS_BASIS_DEVICE` at the top of each script.

To run several back-to-back on one GPU:
`GPU=0 TAG=s_randopt bash admin/run_queue.sh runs/s_randopt/qwen3-0.6b_*.sh`.

### OOD generalization (ablation)

Same question as `vision_exp/runs/ood_generalization`: does a merge picked on
the 4 seen tasks keep (or lose) abilities nobody selected for? Held-out
("unseen") tasks are test-only -- no task vector, no valid pool, never part of
any selection criterion -- and every strategy's final weights get an extra
`unseen_test_avg_score` / `unseen_test_per_task` block next to the seen
numbers (which stay untouched). Unseen tasks: `gsm8k` (1319 test, greedy CoT +
`\boxed{}`) and `mbpp` (450 test, code block executed against the asserts).

Two ways to get those numbers:

1. **Post-hoc, no GD rerun** (the default; the GD cells' weights are already
   saved). `scripts/ood_eval.py` rebuilds every cell of a finished unit --
   pretrained / merged / `coeff_search` from the task vectors, `weight_gd`,
   `weight_gd_lora`, `subspace_gd` from their saved checkpoints (pull with
   `ARCH=qwen3-0.6b METHOD=ta bash admin/download_finetuned.sh`), `bo_search`
   (and any `subspace_gd` cell without a local checkpoint) from
   `coefficients_final` x the method basis -- and scores it on the unseen
   tasks. Writes a copy of the unit to `results/ood_generalization/units/`
   with the `__unseen_gsm8k-mbpp` infix; incremental, saved after every cell.

   ```bash
   bash runs/ood_generalization/qwen3-0.6b_ta.sh     # also _ties, qwen3-1.7b_*
   python scripts/tables/ood_table.py                        # -> results/ood_generalization/{summary.csv,tables.md}
   ```

   `UNSEEN_TEST_SAMPLES` (default 300 in the run scripts) caps each unseen
   pool: both tasks decode up to 512 tokens per example, which is what makes
   an OOD pass cost more than the 4 seen tasks together. `OOD_CELLS` restricts
   to some cells (`coeff_search,weight_gd/avg,...`), `FORCE=1` re-scores.

   **Few-shot vs zero-shot.** Zero-shot, a cell's unseen score mostly says
   whether it still emits `\boxed{}` / a code block at all -- format
   compliance, the thing ifeval already measures (a 0.6b TA diagnostic: the
   0-shot gsm8k gap between merge and GD cells vanished at 4-shot, while
   `merged_coeff1` stayed at 0 either way). So the OOD protocol is
   `--n-shot auto` (`UNSEEN_N_SHOT=auto`): gsm8k 4-shot from its train split,
   mbpp 3-shot from its 50-row valid pool, drawn per test example (seeded by
   row, identical across cells -- one fixed shot set moved a 0.6b cell's
   gsm8k by 16 points on the same rows), inserted as user/assistant turns
   before the query. Such units carry a
   `__fewshot` name infix and `unseen_n_shot` in the JSON; `--n-shot` unset is
   the old zero-shot protocol. `--raw-prompt` (`OOD_RAW_PROMPT=1`) additionally
   scores the pretrained *-Base with plain completion prompts (no chat
   template, which a base model was never trained on) into `unseen_raw_*`
   fields -- the capability reference the `[raw]` row in `tables.md` shows.
   gsm8k is scored flexibly (last `\boxed{}`, else the last number in the
   response): even 4-shot, the 0.6b GD cells reason to the right number and
   stop without boxing it. Zero-shot units written before 2026-09-26 used the
   strict boxed-only scorer.

2. **Live**, for a unit that has yet to run: add `UNSEEN_TASKS=(gsm8k mbpp)`
   (and optionally `UNSEEN_TEST_SAMPLES`) to a `runs/full_comparison/*.sh` config, or
   `--unseen-tasks gsm8k,mbpp` to `scripts/merge_eval.py`. Every strategy then scores
   its final weights on the unseen tasks as it finishes (`directional_sampling`:
   the valid-best draw). Same fields, so `ood_table.py` reads both.

### Cost estimation

Peak GPU memory, wall-clock, and model FLOPs — one warmed-up **action** per
strategy, times a fixed action count (not a full rerun). Same protocol as
`vision_exp`'s section 4, with tokens instead of images as the unit of work:

| strategy | one action | × n_actions |
|---|---|---|
| `coeff_search` | one lambda: build the merge (CPU), load it, generate+score the valid pool | `lambda_steps` (10) |
| `bo_search` | one trial: build W(c) (CPU), load it, generate+score the valid pool | `bo_trials` (50; 20 for TIES/TSVM) |
| `weight_gd_lora` | one train epoch (LoRA adapters) | `gd_epochs` (5) |
| `subspace_gd` | one train epoch | `subspace_epochs` (5) |

The "Weight GD" point is the LoRA variant, which is what the main runs report
at every size (full-parameter `weight_gd` is profile-able with
`--strategies weight_gd`, but has no 4B result to plot against).

Every action is timed end to end, weight assembly included (the CPU merge /
W(c) build and its copy into the model, like `subspace_gd`'s per-step
materialization); FLOPs count model forwards/backwards only. `coeff_search`'s
real sweep also runs a teacher-forced valid-loss forward per lambda; that is
diagnostics (directional_sampling's threshold, logging), not part of picking
λ*, so the profile leaves it out -- at `loss_batch_size=4` its
vocab-sized logits alone peak at 13.6 GiB on Qwen3-0.6B and would swamp the
generate cost being compared.

Inits are the best-scoring cells on the main results (`test_avg_score` in
`results/units`): `weight_gd_lora` is `coeff_best` for TA/DARE and `merged`
for TIES/TSVM (14/16 units); `subspace_gd` is `merged` (6/8 units, and the
only init run at 1.7b -- starting it from λ* lands far below coeff_search);
`bo_search` centers on `coeff_best`. λ* is read from `--coeff-source-dir`
(default `results/`). Override with `--weight-gd-lora-init` /
`--subspace-gd-init` / `--bo-init` (`--weight-gd-init`, default `avg` for
TA/DARE and `merged` for TIES/TSVM, for the opt-in full-parameter run). `bo_search`'s timed
trial is a uniform draw inside the same box the real run searches
(`--bo-radius` / `--bo-lower` / `--bo-basis-device`, main-run defaults), and
its peak memory includes the resident basis when `bo_basis_device` resolves
to cuda. Scores are **not** re-measured — plots read
`test_avg_score` from the main results root, at whichever init the profile
recorded.

FLOPs are `tokens × fwd_flops_per_token`, with the per-token constant read off
`torch.utils.flop_counter.FlopCounterMode` on one teacher-forced forward.
Every position the decoder processes counts (padding included, consistently
with that calibration). Backward ≈ 2× forward, plus 1× more for the gradient
checkpointing both GD strategies train under. This is linear-in-tokens: it
glosses over prefill skipping the lm_head and KV-cached decode attending to a
growing cache.

**Protocol.** The rule for what an implementation may use: standard,
widely used off-the-shelf techniques that any training loop would adopt
(Liger's fused cross-entropy, gradient checkpointing) are in; hand-written,
method-specific memory/time tricks are out. Every strategy runs the plainest layout that fits on one card,
with the same per-step batch (4, the main runs' `gd_batch_size`) for
`generate()` and for the GD micro-batches, no gradient accumulation. The GD
loss is Liger's fused linear cross-entropy (`src/strategies/_common.lm_loss`,
`pip install liger-kernel`): HF's default `model(**batch).loss` materializes
fp32 logits over the 152k vocab, ~3.7 GiB per sequence and ~15 GiB at batch
4, which is an artifact of the loss implementation rather than of any GD
method -- `generate()` already computes only the last position's logits for
the search strategies, so this restores parity rather than tilting it. Same
loss, same gradients, no full-vocab tensor. With M =
one bf16 model copy and N = number of tasks, peak memory should then read
M + activations for `coeff_search` and `bo_search` (merge / W(c) built on the
CPU, `--bo-basis-device cpu`), M + adapters for `weight_gd_lora`, and
(N + 2) M for `subspace_gd` (base + N directions + the model holding W(c),
`--subspace-basis-device cuda`; 1.7B needs a 32 GB card). `generate()` at
batch 4 is still slower than the main runs' `gen_batch_size=32`. Gradient
checkpointing stays on because it is the GD strategies' fixed training recipe
(its recompute is charged in FLOPs). The main runs' own memory-saving choices
(`gen_batch_size=32`, grad accumulation, `auto` basis placement) are
deliberately not reproduced: they trade speed for fit and would make the
memory row reflect card size, not the method.

One script runs Qwen3-0.6B, 1.7B and 4B in order on the same GPU (4B without
`subspace_gd`: its (N + 2) M + grad is 56 GB, and the main results have no
`subspace_gd` cell at 4B either):

```bash
bash runs/cost_estimation/ta.sh            # -> results/cost_estimation/units/
ROWS=memory,time,flops bash runs/cost_estimation/plot.sh
```

The figure draws `coeff_search`, `bo_search` and `weight_gd_lora`;
`subspace_gd` is still profiled but left off by default
(`STRATEGIES=coeff_search,bo_search,subspace_gd,weight_gd_lora bash runs/cost_estimation/plot.sh`
puts it back). `PLOT=1` on `ta.sh` draws the figure at the end;
`ARCHES="<arch>:<base>:<batch>[:<strategies>] ..."` changes which models run;
the optional 4th field overrides `STRATEGIES` for that arch (this is how 4B
drops `subspace_gd`). `ROWS` is a
subset of `memory,time,flops` (memory x is
linear; time and FLOPs are log). The figure is three rows × one column per
model, one marker per strategy. Code accepts any method; only the TA run
script is checked in.

### Calling the runner directly

```bash
python scripts/merge_eval.py \
  --base-model Qwen/Qwen3-0.6B-Base --arch qwen3-0.6b --method ta \
  --tasks bank77,ddxplus,ifeval,usefulness_judge \
  --task-vectors-dir checkpoints/task_vectors \
  --budget full \
  --strategies coeff_search,weight_gd,subspace_gd,baselines \
  --weight-gd-inits pretrained,avg,merged,coeff_best \
  --subspace-gd-inits pretrained,avg,merged,coeff_best
```

Useful flags:

- `--task-vectors-dir` — folder with all checkpoints (default
  `checkpoints/task_vectors`). Each `(task, arch)` resolves to
  `<task>_<arch>/`, descending into the weight-bearing `checkpoint-*` subdir.
- `--task-checkpoints` — explicit comma list (aligned with `--tasks`) that
  overrides resolution when you want exact paths/HF ids.
- `--budget` — `full | medium | low | <k> | <k>_per_class`, applied to the
  valid pool (stratified by label where one exists; flat random sample for
  ifeval, which has none).
- `--strategies` — subset of strategies to run.
- `--weight-gd-inits` / `--subspace-gd-inits` — independent init-point lists for
  `weight_gd` and `subspace_gd` (each a subset of `pretrained,avg,merged,coeff_best`;
  `avg` = coefficient `1/N`, the task-vector average for TA).
- `--ds-inits` / `--ds-alpha` / `--ds-beta` / `--ds-samples` / `--ds-seed` /
  `--ds-select-metric` / `--coeff-source-dir` — `directional_sampling` center
  (`pretrained,avg,merged,coeff_best`), box, draw count, per-draw ranking
  metric (`loss` | `score`), and the results root `coeff_search` is read from.
  See "Directional sampling" above.
- `--multitask-checkpoint` — multi-task ckpt for the `baselines` `upper_bound`
  cell (defaults to auto-resolving `multitask_<t1>+<t2>+...+<tN>_<arch>` inside
  `--task-vectors-dir`; none exist yet, so this cell is silently skipped today).
- `--force` — comma list of strategies to recompute even if already in the JSON.
- `--max-new-tokens` — generation length; default `None` uses each task's own
  budget (`MAX_NEW_TOKENS` in `src/tasks/*.py`, only valid together with each
  task's `ENABLE_THINKING=False` -- see design.md: 16 for bank77/ddxplus/
  usefulness_judge's short answers, 256 for ifeval's free-form
  instruction-following responses). Pass an int to override every task
  uniformly (e.g. for a fast smoke test).
- `--temperature` / `--top-p` / `--gen-batch-size` — generation params used by
  every strategy's scoring pass.
- `--gd-epochs` / `--gd-lr` / `--gd-batch-size` / `--max-seq-length` — `weight_gd` training.
- `--subspace-epochs` / `--subspace-lr` / `--subspace-batch-size` — `subspace_gd` training.
- `--ties-density` — TIES trim density (fraction kept), only used when `--method ties`.
- `--dare-drop-rate` / `--dare-seed` — DARE drop fraction and deterministic
  mask seed, only used when `--method dare`.
- `--save-checkpoints` — save the `weight_gd` / `subspace_gd` finetuned model
  (weights + tokenizer, named by init point) under `checkpoints/finetuned/`.

Results are written to
`results/units/<n>_tasks_<tasks>__<arch>__<method>__<budget>.json`, merging
into any existing file (present cells are skipped unless `--force`d). A
timestamped log is written to `results/logs/`. (`--out-dir` overrides the
`results` root.)
