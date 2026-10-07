"""Model utilities for CLIP/SigLIP task-arithmetic experiments.

Bundles everything needed to load checkpoints, build zero-shot heads, run a
shared multi-task classifier, evaluate, and manipulate task vectors. All
functions work with HuggingFace CLIP/SigLIP models and are self-contained
within ``vision_exp``.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from .classnames import get_classnames
from .templates import get_templates


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #

_VISION_PREFIXES = ("model.vision_model.", "model.visual_projection.")


def _filter_vision_keys(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    has_vision = any(k.startswith("model.vision_model.") for k in raw)
    if has_vision:
        sd = {k.replace("model.", "", 1): v
              for k, v in raw.items()
              if any(k.startswith(pf) for pf in _VISION_PREFIXES)}
        print(f"  {len(sd)} vision parameters (vision_model + visual_projection)")
    else:
        sd = raw
        print(f"  {len(sd)} parameters")
    return sd


def load_state_dict(path_or_model_id: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a model state dict.

    Accepts:
      - A local .bin file (PyTorch state dict saved with torch.save)
      - A local checkpoint directory (contains model.safetensors)
      - A HuggingFace model ID

    For local checkpoints, strips the 'model.' prefix and keeps only
    'vision_model.*' keys (CLIP/SigLIP style) when present.
    """
    p = Path(path_or_model_id).expanduser()
    if p.exists():
        if p.is_file() and p.suffix == ".bin":
            print(f"Loading local .bin checkpoint: {path_or_model_id}")
            raw = torch.load(str(p), map_location=device, weights_only=True)
            return _filter_vision_keys(raw)
        model_file = p / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"Expected {model_file}")
        print(f"Loading local checkpoint: {path_or_model_id}")
        raw = load_file(str(model_file), device=device)
        return _filter_vision_keys(raw)

    print(f"Loading HuggingFace model: {path_or_model_id}")
    model = AutoModel.from_pretrained(path_or_model_id).to(device).eval()
    sd = model.state_dict()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return sd


def save_state_dict(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """Save a vision state dict in the same 'model.'-prefixed safetensors format
    as the training checkpoints so it round-trips through load_state_dict()."""
    os.makedirs(path, exist_ok=True)
    sd = {f"model.{k}": v.contiguous().cpu() for k, v in state_dict.items()}
    save_file(sd, os.path.join(path, "model.safetensors"))


# --------------------------------------------------------------------------- #
# Zero-shot text head construction
# --------------------------------------------------------------------------- #

def extract_text_embeddings(
    model_name: str,
    task_name: str,
    device: str = "cuda",
    class_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Build zero-shot text embeddings for a task using a CLIP/SigLIP model.

    ``class_indices`` keeps a subset of the task's classes, in that order.
    Row ``j`` of the returned matrix is the embedding of original label
    ``class_indices[j]``.

    Returns:
        embeddings: (num_classes, D) normalized class embedding matrix
        logit_scale: temperature scaling factor from the model
        logit_bias: additive bias (SigLIP); 0.0 for CLIP
    """
    classnames = get_classnames(task_name)
    if class_indices is not None:
        classnames = [classnames[i] for i in class_indices]
    templates = get_templates(task_name)
    is_siglip = "siglip" in model_name.lower()
    if is_siglip:
        templates = [lambda c: c]

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    padding = "max_length" if is_siglip else True

    weights = []
    with torch.no_grad():
        for cls in tqdm(classnames, desc=f"Encoding {task_name}"):
            texts = [t(cls) for t in templates]
            inputs = processor(text=texts, return_tensors="pt", padding=padding).to(device)
            emb = model.get_text_features(**inputs)
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(0)
            emb = emb / emb.norm()
            weights.append(emb)

    embeddings = torch.stack(weights)  # (num_classes, D)
    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    logit_bias = model.logit_bias.item() if hasattr(model, "logit_bias") else 0.0

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return embeddings, logit_scale, logit_bias


class ZeroShotCLIPModel:
    """Zero-shot CLIP/SigLIP classifier for a single task.

    Holds a reference to the vision model (so in-place state_dict updates to the
    model immediately affect this wrapper) and pre-computed text embeddings.
    """

    def __init__(
        self,
        vision_model: torch.nn.Module,
        text_embeddings: torch.Tensor,
        logit_scale: float = 100.0,
        logit_bias: float = 0.0,
    ):
        self.vision_model = vision_model
        self.text_embeddings = text_embeddings  # (num_classes, D)
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_emb = self.text_embeddings.to(pixel_values.device)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return image_features @ text_emb.T * self.logit_scale + self.logit_bias


def evaluate(model: ZeroShotCLIPModel, dataloader, device: str) -> Dict[str, float]:
    """Evaluate a ZeroShotCLIPModel on a dataloader. Returns {'accuracy', 'loss'}."""
    model.vision_model.eval()
    total_correct = total_loss = total = 0
    with torch.no_grad():
        for batch in dataloader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixels)
            total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
            total_correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    return {"accuracy": total_correct / total, "loss": total_loss / total}


# --------------------------------------------------------------------------- #
# Multi-task classifier (shared backbone + one frozen zero-shot head per task)
# --------------------------------------------------------------------------- #

class MultiTaskCLIPClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        task_names: list,
        device: str,
        class_indices: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        for param in self.model.parameters():
            param.data = param.data.contiguous()

        if hasattr(self.model, "text_model"):
            for p in self.model.text_model.parameters():
                p.requires_grad = False

        self.heads = nn.ModuleDict()
        for task in task_names:
            print(f"  Building head for {task}...")
            indices = None if class_indices is None else class_indices[task]
            text_emb, logit_scale, _ = extract_text_embeddings(
                model_name, task, device, class_indices=indices,
            )
            num_classes, dim = text_emb.shape
            head = nn.Linear(dim, num_classes, bias=True)
            with torch.no_grad():
                head.weight.copy_(text_emb * logit_scale)
                head.bias.zero_()
            for p in head.parameters():
                p.requires_grad = False
            self.heads[task] = head.to(device)

    def forward(self, pixel_values, task_name, labels=None):
        features = self.model.get_image_features(pixel_values=pixel_values)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = F.normalize(features, dim=-1)
        logits = self.heads[task_name](features)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return logits, loss


def save_multitask_checkpoint(model: MultiTaskCLIPClassifier, path: str) -> None:
    """Save the shared backbone in the training checkpoint format."""
    os.makedirs(path, exist_ok=True)
    sd = {f"model.{k}": v.contiguous() for k, v in model.model.state_dict().items()}
    save_file(sd, os.path.join(path, "model.safetensors"))


# --------------------------------------------------------------------------- #
# Task-vector helpers
# --------------------------------------------------------------------------- #

def shared_float_keys(*sds) -> List[str]:
    """Keys that are floating point in the first sd and present in all sds."""
    first = sds[0]
    return [k for k in first if first[k].is_floating_point() and all(k in sd for sd in sds[1:])]


def task_vector(ft_sd: Dict, base_sd: Dict, keys: List[str]) -> Dict[str, torch.Tensor]:
    """tau = finetuned - base, over the shared float keys."""
    return {k: ft_sd[k].float() - base_sd[k].float() for k in keys}


def sum_task_vectors(vectors: List[Dict[str, torch.Tensor]], keys: List[str]) -> Dict[str, torch.Tensor]:
    out = {k: torch.zeros_like(vectors[0][k]) for k in keys}
    for tau in vectors:
        for k in keys:
            out[k] += tau[k]
    return out


def scale_vector(tau: Dict[str, torch.Tensor], lam: float, keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: lam * tau[k] for k in keys}


def apply_delta(base_sd: Dict, delta: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    """W = base + delta over the given keys (base is copied elsewhere-unchanged)."""
    return {k: base_sd[k].float() + delta[k] for k in keys}


def avg_metrics(results: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    accs = [v["accuracy"] for v in results.values()]
    losses = [v["loss"] for v in results.values()]
    return sum(accs) / len(accs), sum(losses) / len(losses)
