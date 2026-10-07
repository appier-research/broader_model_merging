"""Model utilities for LLM task-arithmetic experiments.

Deliberately generic: checkpoint IO goes through ``AutoModelForCausalLM`` /
``AutoTokenizer`` for both local checkpoint dirs and HF hub ids, so any causal
LM architecture works (not just Qwen3), and sharded/unsharded safetensors are
handled transparently by ``transformers`` -- no manual safetensors parsing
like vision_exp does for CLIP.

Unlike vision_exp's ``MultiTaskCLIPClassifier``, there are no per-task heads
here: an LLM already has one output layer, so "multitask" just means the same
shared decoder weights evaluated/trained across task-specific prompts.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #

def load_tokenizer(path_or_model_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_dtype(name) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype '{name}'. Expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def _attn_implementation(device: str, dtype: torch.dtype) -> str:
    """FlashAttention-2 needs CUDA + half precision; the CPU path here is
    weight surgery only (load_state_dict) and never runs a forward pass, and
    float32 is unsupported -- fall back to sdpa in both cases."""
    if device.startswith("cuda") and dtype in (torch.bfloat16, torch.float16):
        return "flash_attention_2"
    return "sdpa"


def load_causal_lm(path_or_model_id: str, device: str = "cpu", dtype="bfloat16") -> torch.nn.Module:
    torch_dtype = resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        path_or_model_id,
        torch_dtype=torch_dtype,
        attn_implementation=_attn_implementation(device, torch_dtype),
    )
    return model.to(device)


def load_state_dict(path_or_model_id: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a causal LM's state dict via the generic HF loading path (local
    checkpoint dir or hub id, sharded or not)."""
    print(f"Loading causal LM: {path_or_model_id}")
    model = load_causal_lm(path_or_model_id, device=device)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return sd


def save_causal_lm(model: torch.nn.Module, tokenizer, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


# --------------------------------------------------------------------------- #
# Prompt formatting + batched generation
# --------------------------------------------------------------------------- #

class RawPrompt(str):
    """A prompt that is already the exact text the model should continue --
    no chat template, no <think> block. Used for the pretrained-base
    reference in runs/ood_generalization (a *-Base model fed the chat
    template is itself out of distribution; see data.apply_few_shot)."""


def format_prompt(tokenizer, input_, enable_thinking: bool = True) -> str:
    """input_ is either a raw prompt string or a chat ``messages`` list, same
    prompt/messages branching as model-merge-transfer/llm_evals/eval.py.
    ``enable_thinking=False`` renders Qwen3's chat template with an empty,
    closed <think></think> block so the model skips straight to its answer
    (see src/tasks/*.py's ENABLE_THINKING and design.md) -- omitted/True
    matches the tokenizer's own default (thinking left up to the model).
    A :class:`RawPrompt` is returned verbatim."""
    if isinstance(input_, RawPrompt):
        return str(input_)
    messages = [{"role": "user", "content": input_}] if isinstance(input_, str) else input_
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )


def _chunk(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    inputs: List,
    max_new_tokens: int = 256,
    temperature: float = 0.01,
    top_p: float = 0.95,
    device: str = "cuda",
    batch_size: int = 32,
    enable_thinking: bool = True,
) -> List[str]:
    """Batched generate() over a list of prompt strings / chat-messages lists.
    Left-padded so the same input-length slice strips the prompt from every
    row in a batch, regardless of individual prompt length. See
    format_prompt() for ``enable_thinking``."""
    model.eval()
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    do_sample = temperature > 0
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.pad_token_id)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    outputs = []
    for batch in _chunk(inputs, batch_size):
        prompts = [format_prompt(tokenizer, x, enable_thinking=enable_thinking) for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, **gen_kwargs)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    tokenizer.padding_side = prev_padding_side
    return outputs


# --------------------------------------------------------------------------- #
# Task-vector helpers (pure dict[str, Tensor] ops, ported unchanged from
# vision_exp/src/models.py -- architecture agnostic)
# --------------------------------------------------------------------------- #

def shared_float_keys(*sds) -> List[str]:
    first = sds[0]
    return [k for k in first if first[k].is_floating_point() and all(k in sd for sd in sds[1:])]


def task_vector(ft_sd: Dict, base_sd: Dict, keys: List[str]) -> Dict[str, torch.Tensor]:
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
    return {k: base_sd[k].float() + delta[k] for k in keys}


def avg_score(results: Dict[str, float]) -> float:
    return sum(results.values()) / len(results)


def avg_loss(results: Dict[str, Optional[float]]) -> Optional[float]:
    """Like avg_score, but skips tasks with no loss (e.g. ifeval, whose pool
    for this call may have no gold completion -- see tasks/ifeval.py)."""
    present = [v for v in results.values() if v is not None]
    return sum(present) / len(present) if present else None
