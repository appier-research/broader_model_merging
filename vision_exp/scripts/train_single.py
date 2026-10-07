#!/usr/bin/env python3
"""Fine-tune a CLIP vision backbone with a frozen zero-shot text head (single task).

Produces the single-task checkpoints (task vectors) consumed by merge_eval.py.
Not part of the two main results; run from runs/train/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoProcessor,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.data import get_dataloader
from src.models import extract_text_embeddings


class CLIPClassifier(nn.Module):
    """Trainable CLIP vision backbone with a frozen zero-shot text classifier head."""

    def __init__(self, model_name: str, task_name: str, device: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        for param in self.model.parameters():
            param.data = param.data.contiguous()

        if hasattr(self.model, "text_model"):
            for p in self.model.text_model.parameters():
                p.requires_grad = False

        text_emb, logit_scale, _ = extract_text_embeddings(model_name, task_name, device)
        num_classes, dim = text_emb.shape
        self.classifier = nn.Linear(dim, num_classes, bias=True)
        with torch.no_grad():
            self.classifier.weight.copy_(text_emb * logit_scale)
            self.classifier.bias.zero_()
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier = self.classifier.to(device)

    def forward(self, pixel_values, labels=None):
        features = self.model.get_image_features(pixel_values=pixel_values)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = F.normalize(features, dim=-1)
        logits = self.classifier(features)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


class DataLoaderTrainer(Trainer):
    """Trainer that uses pre-built DataLoaders instead of building its own."""

    def __init__(self, *args, train_dataloader, eval_dataloader, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        return self._train_dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        return self._eval_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Save every N steps; defaults to --eval-steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the train/valid split (default: 42)")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model)

    def transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    train_loader = get_dataloader(
        args.task, split="train", transform=transform,
        batch_size=args.batch_size, shuffle=True, seed=args.seed,
    )
    eval_loader = get_dataloader(
        args.task, split="valid", transform=transform,
        batch_size=args.batch_size, shuffle=False, seed=args.seed,
    )

    model = CLIPClassifier(args.model, args.task, device)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        acc = (np.argmax(preds, axis=1) == labels).mean()
        return {"accuracy": float(acc)}

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        max_steps=args.steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=["tensorboard", "wandb"] if args.wandb_project else ["tensorboard"],
    )

    trainer = DataLoaderTrainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        eval_dataset=eval_loader.dataset,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
