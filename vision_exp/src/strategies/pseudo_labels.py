"""Expert pseudo-labels for the unlabeled-data setting.

For each task t the *expert* is the individually finetuned model theta_t
(``ctx.task_sds[i]``). We forward the task's unlabeled data
(``ctx.unlabeled_loaders``) through it once, cache the logits aligned with the
dataset index, and wrap the loader so every batch also carries
``expert_logits``. The GT ``labels`` stay in the batch for diagnostics only;
no trainer reads them when ``gd_label_source != gt``.

Consumers: ``weight_gd`` with ``expert_soft`` / ``expert_hard`` (full weights)
and ``divmerge`` (coefficients) -- the same reference logits DivMerge uses.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data import _collate
from ..models import MultiTaskCLIPClassifier
from .context import StrategyContext

LABEL_SOURCES = ("gt", "expert_soft", "expert_hard")
DIVERGENCES = ("js", "kl")


def soft_divergence(model_logits: torch.Tensor, expert_logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Batch-mean divergence between expert (p) and model (q) predictive distributions.

    kl: KL(p || q) -- standard distillation.  js: Jensen-Shannon (DivMerge default).
    """
    log_p = F.log_softmax(expert_logits, dim=-1)
    log_q = F.log_softmax(model_logits, dim=-1)
    if kind == "kl":
        return F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
    if kind == "js":
        log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - torch.log(torch.tensor(2.0, device=log_p.device))
        kl_pm = (log_p.exp() * (log_p - log_m)).sum(-1)
        kl_qm = (log_q.exp() * (log_q - log_m)).sum(-1)
        return (0.5 * (kl_pm + kl_qm)).mean()
    raise ValueError(f"Unknown divergence '{kind}' (expected {DIVERGENCES})")


def result_key(label_source: str) -> str:
    """Strategy key under ``strategies`` in the unit JSON."""
    if label_source == "gt":
        return "weight_gd"
    if label_source in LABEL_SOURCES:
        return f"weight_gd_{label_source}"
    raise ValueError(f"Unknown gd_label_source '{label_source}' (expected {LABEL_SOURCES})")


class _WithExpertLogits(Dataset):
    def __init__(self, base: Dataset, logits: torch.Tensor):
        assert len(base) == logits.shape[0], (len(base), logits.shape)
        self.base, self.logits = base, logits

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return image, label, self.logits[idx]


def _collate_with_logits(batch):
    images, labels, logits = zip(*batch)
    out = _collate(list(zip(images, labels)))
    out["expert_logits"] = torch.stack(logits)
    return out


@torch.no_grad()
def _expert_logits(model, task: str, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward ``loader.dataset`` in index order; returns (logits [N,C], gt labels [N]) on CPU."""
    ordered = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False,
        num_workers=loader.num_workers, pin_memory=True, collate_fn=_collate,
    )
    model.eval()
    all_logits, all_labels = [], []
    for batch in ordered:
        logits, _ = model(batch["pixel_values"].to(device), task)
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["labels"])
    return torch.cat(all_logits), torch.cat(all_labels)


def get_pseudo_loaders(ctx: StrategyContext) -> Tuple[Dict[str, DataLoader], dict]:
    """Unlabeled loaders whose batches carry ``expert_logits``; cached on ``ctx``.

    ``stats[task]`` = expert accuracy / CE loss on the unlabeled data (i.e. the
    pseudo-label quality; uses GT labels for diagnostics only), plus ``avg``.
    """
    if ctx.pseudo_loaders is not None:
        return ctx.pseudo_loaders, ctx.pseudo_stats

    tasks, device = ctx.tasks, ctx.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("\n=== Expert pseudo-labels: forwarding unlabeled data through each task expert ===")
    model = MultiTaskCLIPClassifier(ctx.base_model, tasks, device, ctx.class_indices)

    loaders, stats = {}, {}
    for i, task in enumerate(tasks):
        expert = {k: ctx.task_sds[i][k].float() for k in ctx.keys}
        model.model.load_state_dict(expert, strict=False)
        src = ctx.unlabeled_loaders[task]
        logits, gt = _expert_logits(model, task, src, device)
        acc = (logits.argmax(-1) == gt).float().mean().item()
        loss = F.cross_entropy(logits, gt).item()
        stats[task] = {"accuracy": acc, "loss": loss, "n": int(gt.numel())}
        logging.info(f"  {task:>14s}: expert acc={acc:.4f}  loss={loss:.4f}  n={gt.numel()}")
        loaders[task] = DataLoader(
            _WithExpertLogits(src.dataset, logits), batch_size=src.batch_size, shuffle=True,
            num_workers=src.num_workers, pin_memory=True, collate_fn=_collate_with_logits,
        )
    stats["avg"] = {
        "accuracy": sum(s["accuracy"] for t, s in stats.items()) / len(tasks),
        "loss": sum(s["loss"] for t, s in stats.items()) / len(tasks),
    }
    logging.info(f"  expert avg acc on unlabeled data={stats['avg']['accuracy']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ctx.pseudo_loaders, ctx.pseudo_stats = loaders, stats
    return loaders, stats


def step_loss(model, batch, task: str, label_source: str, device: str, divergence: str = "js"):
    """One task-batch forward under the configured label source.

    Returns (logits, loss, targets) where ``targets`` are the labels used for
    the loss (GT for ``gt``, expert argmax for ``expert_hard``, expert argmax as
    a diagnostic for ``expert_soft``).
    """
    pixel = batch["pixel_values"].to(device)
    if label_source == "gt":
        labels = batch["labels"].to(device)
        logits, loss = model(pixel, task, labels)
        return logits, loss, labels
    expert = batch["expert_logits"].to(device)
    pseudo = expert.argmax(-1)
    if label_source == "expert_hard":
        logits, loss = model(pixel, task, pseudo)
        return logits, loss, pseudo
    if label_source == "expert_soft":
        logits, _ = model(pixel, task)
        return logits, soft_divergence(logits, expert, divergence), pseudo
    raise ValueError(f"Unknown gd_label_source '{label_source}' (expected {LABEL_SOURCES})")
