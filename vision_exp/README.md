# vision_exp

Model merging (TA, TIES, DARE, TSV-M) on CLIP ViT-B/32, ViT-B/16 and ViT-L/14
over 8 tasks. Code map and protocol details: [`design.md`](design.md).

Run everything from `vision_exp/`. Results go to `results/`; set `GPU=` /
`PYTHON=` as needed.

## 1. Checkpoints

```bash
bash admin/download_task_vectors.sh                    # from GCS (set SRC= or edit the placeholder)
# or train them:
bash runs/train/train_all_single_task_checkpoints.sh     # edit ARCH / MODEL / TASKS inside
```

## 2. Main results

Each run script is one `(arch, method)` unit; rerunning only computes missing
cells. Queue several on one GPU with
`GPU=0 bash admin/run_queue.sh <script>...`.

### Full comparison

Naive merge, coefficient search, weight GD, subspace GD and the baselines per
`(arch, method)`; `*_grid.sh` refines the coefficient grid (subspace upper bound).

```bash
bash runs/full_comparison/vit-b-32_{ta,ties,dare,tsvm}.sh      # also vit-b-16_*, vit-l-14_*
bash runs/full_comparison/vit-b-32_{ties,tsvm}_grid.sh         # refined coefficient grid
```

### BO search

Bayesian optimization over the merge coefficients. Needs the `full_comparison` unit first.

```bash
bash runs/bo_search/vit-b-32_{ta,ties}.sh                      # also vit-b-16_*, vit-l-14_*
```

### Data scaling

Valid budget of 1/2/4/8/10 samples per class, plus full.

```bash
bash runs/data_scaling/vit-b-32_{ta,ties}.sh              # also vit-b-16_*, vit-l-14_{ta,tsvm}
bash runs/data_scaling/plot.sh
```

## 3. Analyses

### Subspace escape

Weight-GD trajectories from points in the task-vector subspace (2 tasks).

```bash
bash runs/subspace_escape/full_2task_ta.sh
bash runs/subspace_escape/plot.sh
```

### S-RandOpt (Directional Sampling)

Random draws around the merge, scored on test.

```bash
bash runs/s_randopt/vit-b-32_{ta,ties,dare}.sh                 # also vit-b-16_*
```

### Cost estimation

Peak memory, wall-clock time and FLOPs per strategy.

```bash
bash runs/cost_estimation/ta.sh
bash runs/cost_estimation/plot.sh
```

### Init scatters

Init score vs weight-GD score, including random-init GD.

```bash
bash runs/init_scatters/vit-b-32.sh                            # also vit-b-16, vit-l-14
METHODS=pretrained,ta,dare,ties,random bash runs/init_scatters/plot.sh
```

### 2-Task Setting

Analyze the 2-task setting.

```bash
bash runs/data_scaling/2task_ta.sh
bash runs/data_scaling/2task_ta_l2_sp.sh  # regularization 
bash runs/data_scaling/plot_2task.sh
```

### Test-time adaptation

AdaMerging, DivMerge and weight GD with task-expert pseudo-labels, all trained
on unlabeled test data (`UNLABELED_SAMPLES` per task; empty = whole test split).

```bash
bash runs/test_time_adaptation/vit-b-32_ta_main.sh   # taskwise coefficients, avg init; also vit-b-16, vit-l-14
bash runs/test_time_adaptation/vit-b-32_ta.sh        # + layerwise coefficients and more inits
```

### Task diversity vs sample size

All 9 tasks but only 53 classes in total, drawn by class seed 0/1/2, across
valid budgets of 1..64 samples per class plus full.

```bash
bash runs/smaller_9task/vit-b-32_ta_cseed{0,1,2}.sh            # budgets 1..64 (independent; run in parallel)
bash runs/smaller_9task/vit-b-32_ta_full.sh                    # full budget, all three class seeds
bash runs/smaller_9task/plot.sh
```