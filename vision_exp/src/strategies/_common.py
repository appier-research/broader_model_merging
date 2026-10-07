"""Evaluation and packing helpers shared across strategies."""

from __future__ import annotations

import os
from typing import Dict, List

import torch

from ..models import avg_metrics, save_state_dict


@torch.no_grad()
def eval_multitask_model(model, task, loader, device) -> Dict[str, float]:
    model.eval()
    correct = total = loss_sum = 0.0
    for batch in loader:
        logits, loss = model(batch["pixel_values"].to(device), task, batch["labels"].to(device))
        n = batch["labels"].size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(-1) == batch["labels"].to(device)).sum().item()
        total += n
    return {"accuracy": correct / total, "loss": loss_sum / total}


def eval_multitask_all(model, loaders, tasks, device) -> Dict[str, Dict[str, float]]:
    return {t: eval_multitask_model(model, t, loaders[t], device) for t in tasks}


def pack_result(valid_res, test_res, **extra) -> dict:
    valid_acc, valid_loss = avg_metrics(valid_res)
    test_acc, test_loss = avg_metrics(test_res)
    out = {
        "valid_avg_acc": valid_acc,
        "valid_avg_loss": valid_loss,
        "test_avg_acc": test_acc,
        "test_avg_loss": test_loss,
        "valid_per_task": valid_res,
        "test_per_task": test_res,
    }
    out.update(extra)
    return out


def save_finetuned(ctx, model_sd: Dict[str, torch.Tensor], keys: List[str], tag: str) -> str:
    """Save a finetuned vision state dict under checkpoint_dir/<tag>. Returns path."""
    path = os.path.join(ctx.checkpoint_dir, tag)
    save_state_dict({k: model_sd[k] for k in keys}, path)
    return path
