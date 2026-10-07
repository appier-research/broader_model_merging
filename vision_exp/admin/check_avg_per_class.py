"""Quick check: average instances per class for the `medium` budget, 9-task scenario.

Mirrors src/experiment.build_budget_loaders (valid pool -> apply_valid_budget),
but skips model/loader construction so it runs fast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import apply_valid_budget, build_valid_pool
from src.classnames import get_classnames

TASKS = [
    "dtd", "eurosat", "fer2013", "food101", "gtsrb",
    "mnist", "resisc45", "stanford-cars", "sun397",
]
BUDGET = "full" #"medium"
SEED = 42

total_used = total_pool = 0
per_class_vals = []
print(f"=== Loaders (budget={BUDGET}) ===")
for task in TASKS:
    pool = build_valid_pool(task, seed=SEED)
    n_classes = len(get_classnames(task))
    _, used, pool_size = apply_valid_budget(pool, BUDGET, seed=SEED)
    avg = used / n_classes
    per_class_vals.append(avg)
    total_used += used
    total_pool += pool_size
    print(f"  {task}: using {used} / {pool_size} valid samples "
          f"({avg:.1f} per class, {n_classes} classes)")

print(f"  total: using {total_used} / {total_pool} valid samples")
print(f"\nAvg instances per class (mean over {len(TASKS)} tasks): "
      f"{sum(per_class_vals) / len(per_class_vals):.2f}")
