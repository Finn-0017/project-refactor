#!/usr/bin/env python
"""Train WHP on precomputed obfuscation passages with deterministic data selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.data import WHPPrecomputedDataset, load_whp_examples
from unlearning_research.modeling import CausalLMWithLoRA, LoRASettings
from unlearning_research.trainer import TrainingSettings, train_whp
from unlearning_research.utils import (
    append_log,
    build_file_manifest,
    copy_if_exists,
    ensure_dir,
    parse_int_list,
    save_json,
    set_seed,
    trainable_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean WHP training")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--names_path", required=True, help="Path to data/WHPplus/whp_names.json")
    parser.add_argument("--obfuscate_passages", required=True)
    parser.add_argument("--selected_ids", required=True)
    parser.add_argument("--lora_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logfile", default=None)
    parser.add_argument("--num_passages", type=int, default=20)
    parser.add_argument("--passage_id", default="-1", help="Comma-separated passage IDs; -1 means nested seed selection")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", default="constant")
    parser.add_argument("--num_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    logfile = args.logfile or str(output_dir / "log.txt")
    lora = LoRASettings.from_json(args.lora_config)

    copy_if_exists(args.selected_ids, output_dir)
    copy_if_exists(args.lora_config, output_dir)

    explicit_passage_ids = parse_int_list(args.passage_id)
    examples, selected_manifest = load_whp_examples(
        names_path=args.names_path,
        selected_ids=args.selected_ids,
        obfuscate_passages_path=args.obfuscate_passages,
        num_passages=args.num_passages,
        seed=args.seed,
        explicit_passage_ids=explicit_passage_ids,
    )

    save_json(vars(args), output_dir / "run_config.json")
    save_json(
        {
            "files": build_file_manifest(
                {
                    "names_path": args.names_path,
                    "obfuscate_passages": args.obfuscate_passages,
                    "selected_ids": args.selected_ids,
                    "lora_config": args.lora_config,
                }
            ),
            "selected_passages": selected_manifest,
        },
        output_dir / "data_manifest.json",
    )

    append_log(f"Loaded {len(examples)} WHP obfuscation examples", logfile)
    for name, meta in selected_manifest.items():
        append_log(f"{name}: passage_ids={meta['passage_ids']}", logfile)

    model = CausalLMWithLoRA(
        args.model_path,
        lora,
        torch_dtype=args.torch_dtype,
    ).to(args.device)
    save_json(trainable_parameter_summary(model), output_dir / "parameter_summary.json")

    dataset = WHPPrecomputedDataset(examples, model.tokenizer)
    settings = TrainingSettings(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_ratio=args.num_warmup_ratio,
        retain_factor=0.0,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        max_grad_norm=args.max_grad_norm,
    )
    train_whp(
        model=model,
        dataset=dataset,
        settings=settings,
        output_dir=output_dir,
        logfile=logfile,
    )


if __name__ == "__main__":
    main()
