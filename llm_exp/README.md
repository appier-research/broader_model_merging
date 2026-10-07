# llm_exp

Model merging (TA, TIES, DARE, TSV-M) on Qwen3-0.6B/1.7B/4B-Base and
Llama-3.2-1B-Instruct over 4 tasks (bank77, ddxplus, ifeval, usefulness_judge);
gsm8k and mbpp are unseen (OOD) test tasks only. Code map and protocol
details: [`design.md`](design.md).

Run everything from `llm_exp/`. Results go to `results/`; set `GPU=` /
`PYTHON=` as needed.

## 1. Checkpoints

Download the task vectors per arch (set `SRC_ROOT=` or edit the placeholder
in the script):

```bash
ARCH=qwen3-0.6b bash admin/download_checkpoints.sh     # also qwen3-1.7b, qwen3-4b, llama3.2-1b
```

Or train them with [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
(install it separately; configs in `configs/train/<arch>/`):

```bash
bash runs/train/train_all_single_task_checkpoints.sh   # ARCHS= / TASKS= to subset
```

## 2. Main results

Each run script is one `(arch, method)` unit; rerunning only computes missing
cells. Queue several on one GPU with
`GPU=0 bash admin/run_queue.sh <script>...`.

### Full comparison

Naive merge, coefficient search, weight GD, weight GD with LoRA, subspace GD and the baselines per
`(arch, method)`; `*_grid.sh` refines the coefficient grid (for subspace upper bound).

```bash
bash runs/full_comparison/qwen3-0.6b_{ta,ties,dare,tsvm}.sh       # also qwen3-1.7b, qwen3-4b, llama3.2-1b
bash runs/full_comparison/qwen3-0.6b_{ta,ties,dare,tsvm}_grid.sh  # refined coefficient grid
```

### BO search

Bayesian optimization over the merge coefficients. Needs the `full_comparison` unit first.

```bash
bash runs/bo_search/qwen3-0.6b_{ta,ties,dare,tsvm}.sh             # also qwen3-1.7b, qwen3-4b, llama3.2-1b
```

## 3. Analyses

### S-RandOpt (Directional Sampling)

Random draws around the merge, scored on test. TA/DARE start from `bo_search`'s best point.

```bash
bash runs/s_randopt/qwen3-0.6b_{ta,ties,dare}.sh                  # also qwen3-1.7b_*, llama3.2-1b_ta
python scripts/plots/plot_ds_test_dist.py --selected top1 --methods ta,dare --layout row
```

### Cost estimation

Peak memory, wall-clock time and FLOPs per strategy.

```bash
bash runs/cost_estimation/ta.sh
bash runs/cost_estimation/plot.sh
```

### Init scatters

Init score vs weight-GD score.

```bash
bash runs/init_scatters/plot.sh
```

### OOD generalization

Score finished units on the unseen gsm8k / mbpp tasks.

```bash
ARCH=qwen3-0.6b METHOD=ta bash admin/download_finetuned.sh        # saved GD weights, if not local
bash runs/ood_generalization/qwen3-0.6b_{ta,ties}.sh              # also qwen3-1.7b_*
python scripts/tables/ood_table.py
```