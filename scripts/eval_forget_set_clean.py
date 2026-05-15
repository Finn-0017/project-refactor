#!/usr/bin/env python
"""Run generation and scoring for one forget set in a single command.

The script evaluates the three probe families used in the unlearning experiments:
open-ended QA, MCQ probes, and Yes/No probes.  It writes raw predictions and aggregate
scores to the same output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.data import load_name_table
from unlearning_research.eval_score import EvaluationSelection, evaluate_and_score_suite, score_suite, write_summary_csv
from unlearning_research.utils import ensure_dir, load_json, load_selected_ids, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified forget-set evaluation and scoring")

    # Model and checkpoint.
    parser.add_argument("--base_model_path", default=None, help="Hugging Face model path. If omitted, read from run_dir/model_config.json when available.")
    parser.add_argument("--run_dir", default=None, help="Experiment directory containing model_config.json, lora_config.json, and checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory or pytorch_model.pt. Required unless --origmodel is set.")
    parser.add_argument("--checkpoint_epoch", type=int, default=1, help="Used only when --checkpoint is omitted and old checkpoint names are present.")
    parser.add_argument("--checkpoint_step", default="final", help="Used only when --checkpoint is omitted and old checkpoint names are present.")
    parser.add_argument("--lora_config", default=None, help="LoRA config JSON. If omitted, tries run_dir/lora_config.json.")
    parser.add_argument("--origmodel", action="store_true", help="Evaluate the base model without loading a LoRA checkpoint.")

    # Forget-set selection.
    parser.add_argument("--selected_ids", required=True, help="Path to config/unlearn_ids*.json or a comma-separated ID list.")
    parser.add_argument("--names_path", default="data/WHPplus/whp_names.json")

    # Probe files. Defaults match the current project naming convention.
    parser.add_argument("--open_file", default="data/WHPplus/whp_unlearn_testset_forget.json")
    parser.add_argument("--mcq_file", default="data/WHPplus/whp_unlearn_testset_forget_mcq.json")
    parser.add_argument("--yesno_gt_file", default=None, help="Prebuilt reference Yes/No questions, e.g. gt_probe_questions.json.")
    parser.add_argument("--yesno_in_file", default=None, help="Prebuilt in-training Yes/No questions, e.g. in_probe_questions.json.")
    parser.add_argument("--yesno_out_file", default=None, help="Prebuilt out-of-training Yes/No questions, e.g. out_probe_questions.json.")
    parser.add_argument("--yesno_all_file", default="data/WHPplus/whp_unlearn_testset_forget_obfuscate_all.json")
    parser.add_argument("--yesno_more_file", default="data/WHPplus/whp_unlearn_testset_forget_obfuscat_more_yesno_all.json")
    parser.add_argument("--obfuscate_samples", default=None, help="Run-specific obfuscate_samples.json. If omitted, tries run_dir/obfuscate_samples.json.")

    # Probe selection.
    parser.add_argument("--no_open", action="store_true")
    parser.add_argument("--no_mcq", action="store_true")
    parser.add_argument("--no_yesno", action="store_true")
    parser.add_argument("--max_new_tokens_open", type=int, default=64)
    parser.add_argument("--max_new_tokens_mcq", type=int, default=32)
    parser.add_argument("--max_new_tokens_yesno", type=int, default=8)

    # Output and runtime.
    parser.add_argument("--output_dir", default=None, help="Defaults to run_dir/eval_forget_set or ./eval_forget_set.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", default="bfloat16")

    # Score-only mode is useful when predictions already exist.
    parser.add_argument("--score_only", default=None, help="Path to predictions.json. Recompute summary without running the model.")
    return parser.parse_args()


def _load_train_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    config_path = run_dir / "model_config.json"
    return load_json(config_path) if config_path.exists() else {}


def _resolve_base_model_path(args: argparse.Namespace, train_config: dict[str, Any]) -> str:
    if args.base_model_path:
        return args.base_model_path
    if "model_path" in train_config:
        return str(train_config["model_path"])
    raise ValueError("Provide --base_model_path or use --run_dir with model_config.json containing model_path.")


def _resolve_lora_config(args: argparse.Namespace, run_dir: Path | None, train_config: dict[str, Any]) -> Path:
    candidates = []
    if args.lora_config:
        candidates.append(Path(args.lora_config))
    if run_dir is not None:
        candidates.append(run_dir / "lora_config.json")
    if train_config.get("lora_config"):
        candidates.append(Path(train_config["lora_config"]))
    for path in candidates:
        if path.exists():
            return path
    raise ValueError("Could not find LoRA config. Pass --lora_config or keep lora_config.json in --run_dir.")


def _resolve_checkpoint(args: argparse.Namespace, run_dir: Path | None) -> Path | None:
    if args.origmodel:
        return None
    candidates = []
    if args.checkpoint:
        candidates.append(Path(args.checkpoint))
    if run_dir is not None:
        candidates.extend(
            [
                run_dir / "checkpoint.final",
                run_dir / f"checkpoint.{args.checkpoint_epoch}.{args.checkpoint_step}",
                run_dir / "pytorch_model.pt",
            ]
        )
    for path in candidates:
        if path.is_dir():
            pt = path / "pytorch_model.pt"
            if pt.exists():
                return pt
        if path.exists() and path.is_file():
            return path
    raise ValueError("Could not find checkpoint. Pass --checkpoint, or use --origmodel.")


def _resolve_output_dir(args: argparse.Namespace, run_dir: Path | None) -> Path:
    if args.output_dir:
        return ensure_dir(args.output_dir)
    if run_dir is not None:
        return ensure_dir(run_dir / "eval_forget_set")
    return ensure_dir("eval_forget_set")


def _build_selection(args: argparse.Namespace) -> EvaluationSelection:
    selected_ids = set(load_selected_ids(args.selected_ids))
    id_to_name = load_name_table(args.names_path) if args.names_path else {}
    selected_names = {id_to_name[x] for x in selected_ids if x in id_to_name}
    return EvaluationSelection(selected_ids=selected_ids, selected_names=selected_names, id_to_name=id_to_name)


def _existing_or_none(path_like: str | None) -> str | None:
    if not path_like:
        return None
    path = Path(path_like)
    return str(path) if path.exists() else None


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args, Path(args.run_dir) if args.run_dir else None)

    if args.score_only:
        predictions = load_json(args.score_only)
        summary = score_suite(predictions)
        save_json(summary, output_dir / "summary.json")
        write_summary_csv(summary, output_dir / "summary.csv")
        print(f"Wrote score-only summary to {output_dir}")
        return

    set_seed(args.seed)
    run_dir = Path(args.run_dir) if args.run_dir else None
    train_config = _load_train_config(run_dir)
    base_model_path = _resolve_base_model_path(args, train_config)
    lora_config_path = _resolve_lora_config(args, run_dir, train_config)
    checkpoint = _resolve_checkpoint(args, run_dir)

    from unlearning_research.modeling import CausalLMWithLoRA, LoRASettings

    lora = LoRASettings.from_json(lora_config_path)
    if args.origmodel:
        lora = LoRASettings(
            lora_rank=lora.lora_rank,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            lora_module=lora.lora_module,
            uselora=False,
        )

    model = CausalLMWithLoRA(base_model_path, lora, torch_dtype=args.torch_dtype).to(args.device)
    if checkpoint is not None:
        model.load_trainable_checkpoint(checkpoint, strict=False)
    model.eval()

    obfuscate_samples = args.obfuscate_samples
    if obfuscate_samples is None and run_dir is not None and (run_dir / "obfuscate_samples.json").exists():
        obfuscate_samples = str(run_dir / "obfuscate_samples.json")

    metadata = {
        "base_model_path": base_model_path,
        "run_dir": str(run_dir) if run_dir else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "lora_config": str(lora_config_path),
        "selected_ids": args.selected_ids,
        "names_path": args.names_path,
        "open_file": args.open_file,
        "mcq_file": args.mcq_file,
        "yesno_gt_file": args.yesno_gt_file,
        "yesno_in_file": args.yesno_in_file,
        "yesno_out_file": args.yesno_out_file,
        "yesno_all_file": args.yesno_all_file,
        "yesno_more_file": args.yesno_more_file,
        "obfuscate_samples": obfuscate_samples,
        "seed": args.seed,
    }

    _, summary = evaluate_and_score_suite(
        model=model,
        output_dir=output_dir,
        selection=_build_selection(args),
        open_file=_existing_or_none(args.open_file),
        mcq_file=_existing_or_none(args.mcq_file),
        yesno_gt_file=_existing_or_none(args.yesno_gt_file),
        yesno_in_file=_existing_or_none(args.yesno_in_file),
        yesno_out_file=_existing_or_none(args.yesno_out_file),
        yesno_all_file=_existing_or_none(args.yesno_all_file),
        yesno_more_file=_existing_or_none(args.yesno_more_file),
        obfuscate_samples_file=_existing_or_none(obfuscate_samples),
        max_new_tokens_open=args.max_new_tokens_open,
        max_new_tokens_mcq=args.max_new_tokens_mcq,
        max_new_tokens_yesno=args.max_new_tokens_yesno,
        run_open=not args.no_open,
        run_mcq=not args.no_mcq,
        run_yesno=not args.no_yesno,
        metadata=metadata,
    )

    print(f"Wrote predictions and scores to {output_dir}")
    for split, metrics in summary.items():
        compact = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        print(f"{split}: {compact}")


if __name__ == "__main__":
    main()
