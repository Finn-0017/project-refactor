#!/usr/bin/env python
"""Train a DF-MCQ model with explicit data and loss settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.data import DFMCQDataset, load_mcq_examples
from unlearning_research.modeling import CausalLMWithLoRA, LoRASettings
from unlearning_research.trainer import TrainingSettings, train_df_mcq
from unlearning_research.utils import (
    append_log,
    build_file_manifest,
    copy_if_exists,
    ensure_dir,
    load_selected_ids,
    save_json,
    set_seed,
    trainable_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean DF-MCQ training")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--selected_ids", required=True)
    parser.add_argument("--lora_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logfile", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--num_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--retain_factor", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument(
        "--flatten_loss",
        choices=["uniform_ce", "model_to_uniform_kl"],
        default="uniform_ce",
        help="Loss used to flatten the forget-set choice distribution.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    logfile = args.logfile or str(output_dir / "log.txt")

    selected_ids = load_selected_ids(args.selected_ids)
    lora = LoRASettings.from_json(args.lora_config)

    copy_if_exists(args.selected_ids, output_dir)
    copy_if_exists(args.lora_config, output_dir)

    save_json(vars(args), output_dir / "run_config.json")
    save_json(
        {
            "files": build_file_manifest(
                {
                    "train_data_path": args.train_data_path,
                    "selected_ids": args.selected_ids,
                    "lora_config": args.lora_config,
                }
            ),
            "selected_ids": selected_ids,
        },
        output_dir / "data_manifest.json",
    )

    append_log(f"Selected target IDs: {selected_ids}", logfile)
    forget_examples, retain_examples = load_mcq_examples(args.train_data_path, selected_ids)
    append_log(
        f"Loaded {len(forget_examples)} forget MCQs and {len(retain_examples)} retain MCQs",
        logfile,
    )

    model = CausalLMWithLoRA(
        args.model_path,
        lora,
        torch_dtype=args.torch_dtype,
    ).to(args.device)
    save_json(trainable_parameter_summary(model), output_dir / "parameter_summary.json")

    dataset = DFMCQDataset(
        forget_examples,
        retain_examples,
        model.tokenizer,
        seed=args.seed,
    )
    settings = TrainingSettings(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_ratio=args.num_warmup_ratio,
        retain_factor=args.retain_factor,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        max_grad_norm=args.max_grad_norm,
    )
    train_df_mcq(
        model=model,
        dataset=dataset,
        settings=settings,
        output_dir=output_dir,
        logfile=logfile,
        flatten_loss=args.flatten_loss,
    )


if __name__ == "__main__":
    main()
