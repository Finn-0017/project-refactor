#!/usr/bin/env python
"""Evaluate a clean DF-MCQ or WHP checkpoint on project-format test files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.data import load_name_table
from unlearning_research.evaluate import evaluate_testfile
from unlearning_research.modeling import CausalLMWithLoRA, LoRASettings
from unlearning_research.utils import ensure_dir, load_selected_ids, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint", default=None, help="Path to pytorch_model.pt or a checkpoint directory")
    parser.add_argument("--lora_config", required=True)
    parser.add_argument("--testfile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--names_path", default=None)
    parser.add_argument("--selected_ids", default=None)
    parser.add_argument("--origmodel", action="store_true")
    parser.add_argument("--mcq", action="store_true", help="Force MCQ evaluation")
    parser.add_argument("--yes_no", action="store_true", help="Use Yes/No prompt for open rows")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", default="bfloat16")
    return parser.parse_args()


def resolve_checkpoint(path: str | None) -> Path | None:
    if path is None:
        return None
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "pytorch_model.pt"
    return checkpoint


def selected_names_from_args(args: argparse.Namespace) -> set[str] | None:
    if not args.selected_ids:
        return None
    selected_ids = load_selected_ids(args.selected_ids)
    names = set(selected_ids)
    if args.names_path:
        id_to_name = load_name_table(args.names_path)
        names.update(id_to_name[x] for x in selected_ids if x in id_to_name)
    return names


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outfile = Path(args.outfile)
    ensure_dir(outfile.parent)

    lora = LoRASettings.from_json(args.lora_config)
    if args.origmodel:
        lora = LoRASettings(
            lora_rank=lora.lora_rank,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            lora_module=lora.lora_module,
            uselora=False,
        )

    model = CausalLMWithLoRA(
        args.model_path,
        lora,
        torch_dtype=args.torch_dtype,
    ).to(args.device)

    checkpoint = resolve_checkpoint(args.checkpoint)
    if not args.origmodel:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("A valid --checkpoint is required unless --origmodel is set.")
        model.load_trainable_checkpoint(checkpoint, strict=False)
    model.eval()

    results = evaluate_testfile(
        model=model,
        testfile=args.testfile,
        outfile=outfile,
        selected_names=selected_names_from_args(args),
        mcq=True if args.mcq else None,
        yes_no=args.yes_no,
    )
    save_json(
        {
            "num_people": len(results),
            "num_items": sum(len(v) for v in results.values()),
            "outfile": str(outfile),
        },
        outfile.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
