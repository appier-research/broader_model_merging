"""Dataset loading utilities for image classification tasks."""

import random
from collections import defaultdict
from typing import Literal, Optional, Callable, Tuple

from datasets import Dataset as HFDataset
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset
import torch


DATASET_MAPPING = {
    "imagenet": "imagenet-1k",
    "imagenet-1k": "imagenet-1k",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "stanford-cars": "tanganke/stanford_cars",
    "flowers102": "dpdl-benchmark/oxford_flowers102",
    "food101": "ethz/food101",
    "sun397": "tanganke/sun397",
    "eurosat": "tanganke/eurosat",
    "resisc45": "tanganke/resisc45",
    "dtd": "tanganke/dtd",
    "fer2013": "clip-benchmark/wds_fer2013",
    "gtsrb": "tanganke/gtsrb",
    "mnist": "ylecun/mnist",
    "fashion-mnist": "fashion_mnist",
    "svhn": "svhn",
    "stl10": "tanganke/stl10",
    "rendered-sst2": "nateraw/rendered-sst2",
    "pcam": "1aurent/PatchCamelyon",
}

_IMAGE_KEYS = ["image", "img", "images", "pixel_values", "jpg", "png"]
_LABEL_KEYS = ["label", "labels", "fine_label", "coarse_label", "cls", "class", "category"]


def _find_key(sample: dict, candidates: list) -> Optional[str]:
    for k in candidates:
        if k in sample:
            return k
    return None


class ImageDataset(Dataset):
    def __init__(self, hf_dataset, transform: Optional[Callable] = None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        sample = hf_dataset[0]
        self.image_key = _find_key(sample, _IMAGE_KEYS)
        self.label_key = _find_key(sample, _LABEL_KEYS)
        if not self.image_key or not self.label_key:
            raise ValueError(f"Cannot detect image/label keys. Sample keys: {list(sample.keys())}")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample[self.image_key]
        label = sample[self.label_key]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _collate(batch):
    images, labels = zip(*batch)
    return {
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


BudgetMode = Literal["low", "medium", "full"]


def parse_budget(budget: str):
    """Normalize a CLI budget string into an internal spec.

    Accepts:
      "full" / "medium" / "low"          -> the named modes
      "<k>" or "<k>_per_class"           -> exactly k samples per class
    Returns either one of {"low","medium","full"} or an int k (per class).
    """
    if isinstance(budget, int):
        return budget
    b = str(budget).strip().lower()
    if b in ("low", "medium", "full"):
        return b
    if b.endswith("_per_class"):
        b = b[: -len("_per_class")]
    if b.isdigit():
        return int(b)
    raise ValueError(
        f"Unknown budget '{budget}'. Expected low, medium, full, or an integer per-class count."
    )


def _dataset_kwargs(task_name: str) -> dict:
    if task_name == "svhn":
        return {"name": "cropped_digits"}
    if task_name == "cifar100":
        return {"name": "cifar100"}
    return {}


def _load_hf_split(task_name: str, split: str, shuffle: bool = False, seed: int = 42) -> HFDataset:
    if task_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown task '{task_name}'. Available: {sorted(DATASET_MAPPING)}")
    path = DATASET_MAPPING[task_name]
    kwargs = _dataset_kwargs(task_name)
    try:
        ds = hf_load_dataset(path, split=split, **kwargs)
    except ValueError as e:
        if "Unknown split" in str(e) and split == "test":
            for alt in ["validation", "valid", "val"]:
                try:
                    ds = hf_load_dataset(path, split=alt, **kwargs)
                    break
                except Exception:
                    continue
            else:
                raise
        else:
            raise
    if shuffle:
        ds = ds.shuffle(seed=seed)
    return ds


def _label_key(dataset: HFDataset) -> str:
    key = _find_key(dataset[0], _LABEL_KEYS)
    if not key:
        raise ValueError(f"Cannot detect label key. Sample keys: {list(dataset[0].keys())}")
    return key


def restrict_classes(dataset: HFDataset, kept_ids: list) -> HFDataset:
    """Keep rows whose label is in ``kept_ids`` and remap those labels to 0..k-1.

    ``kept_ids`` is the original label order. Remapped label ``j`` is ``kept_ids[j]``,
    which is also the order the zero-shot head is built in.
    """
    if not kept_ids:
        raise ValueError("kept_ids is empty")
    label_key = _label_key(dataset)
    index_of = {int(old): new for new, old in enumerate(kept_ids)}
    labels = [int(x) for x in dataset[label_key]]
    keep = [i for i, lab in enumerate(labels) if lab in index_of]
    if not keep:
        raise ValueError(f"no rows left after keeping classes {list(kept_ids)}")
    new_labels = [index_of[labels[i]] for i in keep]
    return dataset.select(keep).remove_columns([label_key]).add_column(label_key, new_labels)


def _stratified_fraction(indices_by_label: dict, fraction: float, rng: random.Random) -> list:
    selected = []
    for indices in indices_by_label.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = max(1, round(len(indices) * fraction)) if fraction < 1.0 else len(indices)
        selected.extend(shuffled[:n])
    return sorted(selected)


def _train_valid_split(
    task_name: str, valid_fraction: float = 0.1, seed: int = 42
) -> Tuple[HFDataset, HFDataset]:
    """Return non-overlapping (train, valid) datasets split from the train split."""
    train = _load_hf_split(task_name, split="train", shuffle=False)
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    n_valid = max(1, round(len(indices) * valid_fraction))
    valid_indices = sorted(indices[:n_valid])
    train_indices = sorted(indices[n_valid:])
    return train.select(train_indices), train.select(valid_indices)


def build_valid_pool(task_name: str, valid_fraction: float = 0.1, seed: int = 42) -> HFDataset:
    """Hold out a stratified fraction of the training split as the valid pool."""
    _, valid = _train_valid_split(task_name, valid_fraction=valid_fraction, seed=seed)
    return valid


def _select_per_class(by_label: dict, k: int, rng: random.Random) -> list:
    """Pick up to k samples per class (deterministic given rng)."""
    selected = []
    for label, indices in by_label.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        if len(shuffled) < k:
            print(f"  warning: class {label} has {len(shuffled)} valid samples (< {k})")
        selected.extend(shuffled[: min(k, len(shuffled))])
    return sorted(selected)


def apply_valid_budget(
    valid_pool: HFDataset,
    mode,
    per_class_low: int = 10,
    seed: int = 42,
) -> Tuple[HFDataset, int, int]:
    """Subsample the valid pool according to the budget.

    ``mode`` may be one of "low" / "medium" / "full", or an integer (or the
    output of :func:`parse_budget`) giving an exact per-class sample count.
    """
    mode = parse_budget(mode)
    label_key = _label_key(valid_pool)
    by_label = defaultdict(list)
    for i in range(len(valid_pool)):
        by_label[valid_pool[i][label_key]].append(i)

    rng = random.Random(seed)
    if isinstance(mode, int):
        selected = _select_per_class(by_label, mode, rng)
    elif mode == "low":
        selected = _select_per_class(by_label, per_class_low, rng)
    elif mode == "medium":
        selected = _stratified_fraction(by_label, 0.5, rng)
    elif mode == "full":
        selected = list(range(len(valid_pool)))
    else:
        raise ValueError(f"Unknown budget mode '{mode}'. Expected low, medium, full, or an integer.")
    return valid_pool.select(sorted(selected)), len(selected), len(valid_pool)


def load_hf_dataset(task_name: str, split: str = "test", num_samples: Optional[int] = None):
    """Load a HuggingFace dataset for the given task name."""
    ds = _load_hf_split(task_name, split=split, shuffle=True, seed=42)
    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    return ds


def get_dataloader(
    task_name: str,
    split: str = "test",
    transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    num_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Return a DataLoader for the given split.

    split:
      "train" — 90% of the raw train split (non-overlapping with "valid")
      "valid" — stratified 10% of the raw train split
      "test"  — the dataset's test split
    seed controls the train/valid partition and is ignored for "test".
    """
    if split == "valid":
        _, ds = _train_valid_split(task_name, seed=seed)
    elif split == "train":
        ds, _ = _train_valid_split(task_name, seed=seed)
        if num_samples is not None and num_samples < len(ds):
            ds = ds.select(range(num_samples))
    else:
        ds = load_hf_dataset(task_name, split=split, num_samples=num_samples)
    return get_dataloader_from_dataset(
        ds, transform=transform, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle,
    )


def get_dataloader_from_dataset(
    dataset: HFDataset,
    transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        ImageDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
