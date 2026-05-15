"""Training loops for the clean DF-MCQ and WHP paths."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .choice import choice_logits, normalized_choice_probs
from .data import DFMCQDataset, WHPPrecomputedDataset, collate_dfmcq, collate_whp
from .losses import (
    causal_lm_cross_entropy,
    gradient_l2_norm,
    model_to_uniform_kl,
    reference_choice_cross_entropy,
    uniform_choice_cross_entropy,
)
from .modeling import CausalLMWithLoRA
from .utils import append_log, ensure_dir, move_batch_to_device, save_json


@dataclass(frozen=True)
class TrainingSettings:
    """Shared optimization settings."""

    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    num_warmup_ratio: float = 0.05
    log_interval: int = 50
    save_interval: int = 0
    retain_factor: float = 0.0
    max_grad_norm: float | None = None


def _build_optimizer(model: torch.nn.Module, settings: TrainingSettings) -> AdamW:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    grouped = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(key in n for key in no_decay)
            ],
            "weight_decay": settings.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(key in n for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(grouped, lr=settings.learning_rate)


def _build_scheduler(optimizer, dataloader: DataLoader, settings: TrainingSettings):
    update_steps_per_epoch = math.ceil(
        len(dataloader) / max(1, settings.gradient_accumulation_steps)
    )
    total_update_steps = update_steps_per_epoch * settings.num_train_epochs
    warmup_steps = int(settings.num_warmup_ratio * total_update_steps)
    return get_scheduler(
        name=settings.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )


def train_df_mcq(
    *,
    model: CausalLMWithLoRA,
    dataset: DFMCQDataset,
    settings: TrainingSettings,
    output_dir: str | Path,
    logfile: str | Path | None = None,
    flatten_loss: str = "uniform_ce",
) -> None:
    """Train with DF-MCQ.

    Forget loss: flatten the answer-letter distribution on target-person MCQs.
    Retain loss: match the base model's answer-letter distribution on non-target MCQs.
    """

    output_dir = ensure_dir(output_dir)
    pad_id = model.tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_dfmcq(batch, pad_id),
    )
    optimizer = _build_optimizer(model, settings)
    scheduler = _build_scheduler(optimizer, loader, settings)

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for epoch in range(settings.num_train_epochs):
        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, model.device)

            forget_ids = batch["forget_input_ids"]
            retain_ids = batch["retain_input_ids"]
            forget_mask = batch["forget_attention_mask"]
            retain_mask = batch["retain_attention_mask"]
            forget_outputs = model(forget_ids, attention_mask=forget_mask)
            forget_choice_logits = choice_logits(
                forget_outputs.logits,
                forget_ids,
                model.tokenizer,
                attention_mask=forget_mask,
                model_path=model.model_path,
            )
            if flatten_loss == "model_to_uniform_kl":
                loss_forget = model_to_uniform_kl(forget_choice_logits)
            elif flatten_loss == "uniform_ce":
                loss_forget = uniform_choice_cross_entropy(forget_choice_logits)
            else:
                raise ValueError(f"Unknown flatten loss: {flatten_loss}")

            if settings.retain_factor > 0:
                with torch.no_grad():
                    base_outputs = model.base_model(retain_ids, attention_mask=retain_mask)
                    base_choice_logits = choice_logits(
                        base_outputs.logits,
                        retain_ids,
                        model.tokenizer,
                        attention_mask=retain_mask,
                        model_path=model.model_path,
                    )
                    base_choice_probs = normalized_choice_probs(base_choice_logits)
                retain_outputs = model(retain_ids, attention_mask=retain_mask)
                retain_choice_logits = choice_logits(
                    retain_outputs.logits,
                    retain_ids,
                    model.tokenizer,
                    attention_mask=retain_mask,
                    model_path=model.model_path,
                )
                loss_retain = reference_choice_cross_entropy(
                    retain_choice_logits,
                    base_choice_probs,
                )
            else:
                loss_retain = torch.zeros((), device=model.device)

            loss = loss_forget + settings.retain_factor * loss_retain
            scaled_loss = loss / settings.gradient_accumulation_steps
            scaled_loss.backward()

            should_step = (step + 1) % settings.gradient_accumulation_steps == 0
            if should_step:
                if settings.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), settings.max_grad_norm)
                grad_norm = gradient_l2_norm(model)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % settings.log_interval == 0:
                    elapsed = time.time() - start
                    append_log(
                        (
                            f"epoch={epoch} step={global_step} "
                            f"loss={loss.item():.6f} "
                            f"forget={loss_forget.item():.6f} "
                            f"retain={loss_retain.item():.6f} "
                            f"grad_norm={grad_norm:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e} "
                            f"elapsed={elapsed:.1f}s"
                        ),
                        logfile,
                    )

                if settings.save_interval and global_step % settings.save_interval == 0:
                    model.save_trainable_checkpoint(output_dir / f"checkpoint.step{global_step}")

    model.save_trainable_checkpoint(output_dir / "checkpoint.final")
    save_json({"global_step": global_step}, output_dir / "training_state.json")


def train_whp(
    *,
    model: CausalLMWithLoRA,
    dataset: WHPPrecomputedDataset,
    settings: TrainingSettings,
    output_dir: str | Path,
    logfile: str | Path | None = None,
) -> None:
    """Train WHP using precomputed obfuscation passages.

    This clean path uses standard next-token cross-entropy on obfuscation passages. It is
    appropriate when the passage text is already available and the experiment is focused
    on data amount, LoRA capacity, or seed control.
    """

    output_dir = ensure_dir(output_dir)
    pad_id = model.tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_whp(batch, pad_id),
    )
    optimizer = _build_optimizer(model, settings)
    scheduler = _build_scheduler(optimizer, loader, settings)

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for epoch in range(settings.num_train_epochs):
        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, model.device)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = causal_lm_cross_entropy(outputs.logits, labels)
            scaled_loss = loss / settings.gradient_accumulation_steps
            scaled_loss.backward()

            should_step = (step + 1) % settings.gradient_accumulation_steps == 0
            if should_step:
                if settings.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), settings.max_grad_norm)
                grad_norm = gradient_l2_norm(model)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % settings.log_interval == 0:
                    elapsed = time.time() - start
                    ppl = math.exp(min(loss.item(), 20.0))
                    append_log(
                        (
                            f"epoch={epoch} step={global_step} "
                            f"loss={loss.item():.6f} ppl={ppl:.3f} "
                            f"grad_norm={grad_norm:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e} "
                            f"elapsed={elapsed:.1f}s"
                        ),
                        logfile,
                    )

                if settings.save_interval and global_step % settings.save_interval == 0:
                    model.save_trainable_checkpoint(output_dir / f"checkpoint.step{global_step}")

    model.save_trainable_checkpoint(output_dir / "checkpoint.final")
    save_json({"global_step": global_step}, output_dir / "training_state.json")
