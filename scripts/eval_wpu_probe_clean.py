#!/usr/bin/env python
"""Evaluate one WPU/WHP forget set on all probe files in one command.

This is the main evaluation entry point for the current experiments.  It evaluates the
same model checkpoint on open-ended QA, MCQ probes, and Yes/No probes, then writes both
raw predictions and aggregate scores.

The default file names match the current `data/whp_probe/` layout:

- forget open-ended QA,
- forget MCQ,
- forget Yes/No reference probes,
- forget Yes/No false controls,
- hard-retain open-ended QA,
- hard-retain MCQ,
- hard-retain Yes/No controls,
- retain open-ended QA,
- retain Yes/No controls.

Each run evaluates only one forget set.  Pass `--selected_ids config/unlearn_idsX.json`
to select the two target people for that set.  Retain splits are intentionally not
filtered by the forget-set names.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.data import load_name_table
from unlearning_research.eval_score import (
    EvaluationSelection,
    evaluate_split,
    flatten_people,
    score_named_split,
    write_summary_csv,
)
from unlearning_research.utils import ensure_dir, load_json, load_selected_ids, save_json, set_seed


@dataclass(frozen=True)
class ProbeSpec:
    """One probe split to evaluate and score."""

    split: str
    probe_type: str
    filename: str
    scope: str

    @property
    def requires_selection(self) -> bool:
        return self.scope in {"forget", "hardretain"}


DEFAULT_PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec("forget_open", "open", "whp_unlearn_testset_forget.json", "forget"),
    ProbeSpec("forget_mcq", "mcq", "whp_unlearn_testset_forget_mcq.json", "forget"),
    ProbeSpec("forget_yesno", "yes_no", "whp_unlearn_testset_forget_yesno.json", "forget"),
    ProbeSpec("forget_yesno_false_control", "yes_no", "whp_unlearn_testset_forget_yesno_false_control.json", "forget"),
    ProbeSpec("hardretain_open", "open", "whp_unlearn_testset_hardretain.json", "hardretain"),
    ProbeSpec("hardretain_mcq", "mcq", "whp_unlearn_testset_hardretain_mcq.json", "hardretain"),
    ProbeSpec("hardretain_yesno", "yes_no", "whp_unlearn_testset_hardretain_yesno.json", "hardretain"),
    ProbeSpec("retain_open", "open", "whp_unlearn_testset_retain.json", "retain"),
    ProbeSpec("retain_yesno", "yes_no", "whp_unlearn_testset_retain_yesno.json", "retain"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified WPU/WHP probe evaluation for one forget set")

    # Model and checkpoint.
    parser.add_argument("--base_model_path", default=None, help="Hugging Face model path. If omitted, read from run_dir/model_config.json when available.")
    parser.add_argument("--run_dir", default=None, help="Experiment directory containing model_config.json, lora_config.json, and checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory or pytorch_model.pt. Required unless --origmodel is set.")
    parser.add_argument("--checkpoint_epoch", type=int, default=1, help="Used only when --checkpoint is omitted and old checkpoint names are present.")
    parser.add_argument("--checkpoint_step", default="final", help="Used only when --checkpoint is omitted and old checkpoint names are present.")
    parser.add_argument("--lora_config", default=None, help="LoRA config JSON. If omitted, tries run_dir/lora_config.json.")
    parser.add_argument("--origmodel", action="store_true", help="Evaluate the base model without loading a LoRA checkpoint.")

    # Dataset selection.
    parser.add_argument("--data_dir", default="data/whp_probe", help="Directory containing the whp_unlearn_testset_*.json probe files.")
    parser.add_argument("--selected_ids", required=True, help="Path to config/unlearn_ids*.json or a comma-separated ID list.")
    parser.add_argument("--names_path", default="data/WHPplus/whp_names.json", help="Maps selected IDs to names. If absent, file keys are matched directly by name.")

    # Runtime.
    parser.add_argument("--run_name", default=None, help="Name written into Brian-style table outputs. Defaults to run_dir name or 'model'.")
    parser.add_argument("--output_dir", default=None, help="Defaults to run_dir/eval_wpu_probe or ./eval_wpu_probe.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens_open", type=int, default=64)
    parser.add_argument("--max_new_tokens_mcq", type=int, default=8)
    parser.add_argument("--max_new_tokens_yesno", type=int, default=8)

    # Split switches.
    parser.add_argument("--no_open", action="store_true")
    parser.add_argument("--no_mcq", action="store_true")
    parser.add_argument("--no_yesno", action="store_true")
    parser.add_argument("--skip_missing", action="store_true", default=True, help="Skip probe files that do not exist. Enabled by default.")

    # Score-only mode.
    parser.add_argument("--score_only", default=None, help="Path to predictions.json. Recompute summaries without running the model.")
    return parser.parse_args()


def _load_train_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    path = run_dir / "model_config.json"
    return load_json(path) if path.exists() else {}


def _resolve_base_model_path(args: argparse.Namespace, train_config: dict[str, Any]) -> str:
    if args.base_model_path:
        return args.base_model_path
    if "model_path" in train_config:
        return str(train_config["model_path"])
    raise ValueError("Provide --base_model_path or use --run_dir with model_config.json containing model_path.")


def _resolve_lora_config(args: argparse.Namespace, run_dir: Path | None, train_config: dict[str, Any]) -> Path:
    candidates: list[Path] = []
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
    candidates: list[Path] = []
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
        if path.is_dir() and (path / "pytorch_model.pt").exists():
            return path / "pytorch_model.pt"
        if path.is_file():
            return path
    raise ValueError("Could not find checkpoint. Pass --checkpoint, or use --origmodel.")


def _resolve_output_dir(args: argparse.Namespace, run_dir: Path | None) -> Path:
    if args.output_dir:
        return ensure_dir(args.output_dir)
    if run_dir is not None:
        return ensure_dir(run_dir / "eval_wpu_probe")
    return ensure_dir("eval_wpu_probe")


def _build_selection(args: argparse.Namespace) -> EvaluationSelection:
    selected_ids = set(load_selected_ids(args.selected_ids))
    id_to_name = load_name_table(args.names_path) if args.names_path and Path(args.names_path).exists() else {}
    selected_names = {id_to_name[x] for x in selected_ids if x in id_to_name}
    # Also allow selected_ids to directly contain names for quick local evaluation.
    selected_names.update(x for x in selected_ids if not x.isdigit())
    return EvaluationSelection(selected_ids=selected_ids, selected_names=selected_names, id_to_name=id_to_name)


def _should_run(spec: ProbeSpec, args: argparse.Namespace) -> bool:
    if spec.probe_type == "open" and args.no_open:
        return False
    if spec.probe_type == "mcq" and args.no_mcq:
        return False
    if spec.probe_type == "yes_no" and args.no_yesno:
        return False
    return True


def _max_tokens_for(spec: ProbeSpec, args: argparse.Namespace) -> int:
    if spec.probe_type == "open":
        return args.max_new_tokens_open
    if spec.probe_type == "mcq":
        return args.max_new_tokens_mcq
    return args.max_new_tokens_yesno


def _load_split_rows(spec: ProbeSpec, args: argparse.Namespace, selection: EvaluationSelection) -> dict[str, list[dict[str, Any]]]:
    path = Path(args.data_dir) / spec.filename
    if not path.exists():
        if args.skip_missing:
            print(f"[skip] missing {path}", flush=True)
            return {}
        raise FileNotFoundError(path)
    data = load_json(path)
    # Forget and hard-retain splits are associated with the current forget set. Retain
    # splits intentionally use all rows because they are global utility checks.
    return flatten_people(data, selection if spec.requires_selection else None)


def _score_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    split_types = predictions.get("metadata", {}).get("split_types", {})
    for split_name, payload in predictions.items():
        if split_name == "metadata" or not isinstance(payload, dict):
            continue
        probe_type = split_types.get(split_name)
        if probe_type is None:
            if "mcq" in split_name:
                probe_type = "mcq"
            elif "yesno" in split_name or "yes_no" in split_name:
                probe_type = "yes_no"
            else:
                probe_type = "open"
        summary[split_name] = score_named_split(probe_type, payload)
    return summary


def _write_brian_style_tables(summary: dict[str, Any], output_dir: Path, run_name: str) -> None:
    """Write compact tables matching the paper-style reporting layout."""

    def get(split: str, key: str) -> Any:
        return summary.get(split, {}).get(key)

    open_row = {
        "run_name": run_name,
        "forget_rougeL_recall": get("forget_open", "rougeL_recall"),
        "retain_rougeL_recall": get("retain_open", "rougeL_recall"),
        "hardretain_rougeL_recall": get("hardretain_open", "rougeL_recall"),
        "forget_refusal_rate": get("forget_open", "refusal_rate"),
        "forget_n": get("forget_open", "n"),
        "retain_n": get("retain_open", "n"),
        "hardretain_n": get("hardretain_open", "n"),
    }
    mcq_row = {
        "run_name": run_name,
        # Main MCQ accuracy uses the parsed letter from greedy generation.
        "forget_accuracy": get("forget_mcq", "accuracy"),
        "forget_entropy": get("forget_mcq", "entropy"),
        "forget_normalized_entropy": get("forget_mcq", "normalized_entropy"),
        "forget_p_correct": get("forget_mcq", "p_correct"),
        "forget_p_obfuscation": get("forget_mcq", "p_obfuscation"),
        "forget_letter_extraction_rate": get("forget_mcq", "letter_extraction_rate"),
        # Diagnostic accuracy from the forced next-token distribution.
        "forget_accuracy_distribution": get("forget_mcq", "accuracy_distribution"),
        "hardretain_accuracy": get("hardretain_mcq", "accuracy"),
        "hardretain_entropy": get("hardretain_mcq", "entropy"),
        "hardretain_normalized_entropy": get("hardretain_mcq", "normalized_entropy"),
        "hardretain_p_correct": get("hardretain_mcq", "p_correct"),
        "hardretain_letter_extraction_rate": get("hardretain_mcq", "letter_extraction_rate"),
        "hardretain_accuracy_distribution": get("hardretain_mcq", "accuracy_distribution"),
        "forget_n": get("forget_mcq", "n"),
        "hardretain_n": get("hardretain_mcq", "n"),
    }
    yesno_row = {
        "run_name": run_name,
        "forget_ref_accuracy": get("forget_yesno", "accuracy_distribution"),
        "forget_ref_entropy": get("forget_yesno", "entropy"),
        "forget_ref_yes_rate": get("forget_yesno", "yes_rate"),
        "forget_false_accuracy": get("forget_yesno_false_control", "accuracy_distribution"),
        "forget_false_entropy": get("forget_yesno_false_control", "entropy"),
        "forget_false_yes_rate": get("forget_yesno_false_control", "yes_rate"),
        "retain_accuracy": get("retain_yesno", "accuracy_distribution"),
        "retain_entropy": get("retain_yesno", "entropy"),
        "retain_yes_rate": get("retain_yesno", "yes_rate"),
        "hardretain_accuracy": get("hardretain_yesno", "accuracy_distribution"),
        "hardretain_entropy": get("hardretain_yesno", "entropy"),
        "hardretain_yes_rate": get("hardretain_yesno", "yes_rate"),
        "forget_ref_n": get("forget_yesno", "n"),
        "forget_false_n": get("forget_yesno_false_control", "n"),
        "retain_n": get("retain_yesno", "n"),
        "hardretain_n": get("hardretain_yesno", "n"),
    }

    for filename, row in [
        ("brian_table_open.csv", open_row),
        ("brian_table_mcq.csv", mcq_row),
        ("brian_table_yesno.csv", yesno_row),
    ]:
        path = output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


def _run_name(args: argparse.Namespace, run_dir: Path | None) -> str:
    if args.run_name:
        return args.run_name
    if run_dir is not None:
        return run_dir.name
    return "model"


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args, Path(args.run_dir) if args.run_dir else None)

    if args.score_only:
        predictions = load_json(args.score_only)
        summary = _score_predictions(predictions)
        save_json(summary, output_dir / "summary.json")
        write_summary_csv(summary, output_dir / "summary.csv")
        _write_brian_style_tables(summary, output_dir, _run_name(args, Path(args.run_dir) if args.run_dir else None))
        print(f"Wrote score-only summaries to {output_dir}")
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

    selection = _build_selection(args)
    predictions: dict[str, Any] = {
        "metadata": {
            "base_model_path": base_model_path,
            "run_dir": str(run_dir) if run_dir else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "lora_config": str(lora_config_path),
            "data_dir": args.data_dir,
            "selected_ids": sorted(selection.selected_ids),
            "selected_names": sorted(selection.selected_names),
            "split_types": {},
        }
    }

    for spec in DEFAULT_PROBES:
        if not _should_run(spec, args):
            continue
        rows = _load_split_rows(spec, args, selection)
        if not rows:
            continue
        predictions["metadata"]["split_types"][spec.split] = spec.probe_type
        predictions[spec.split] = evaluate_split(
            model,
            rows,
            probe_type=spec.probe_type,
            max_new_tokens=_max_tokens_for(spec, args),
            desc=spec.split,
        )

    summary = _score_predictions(predictions)
    save_json(predictions, output_dir / "predictions.json")
    save_json(summary, output_dir / "summary.json")
    write_summary_csv(summary, output_dir / "summary.csv")
    _write_brian_style_tables(summary, output_dir, _run_name(args, run_dir))
    print(f"Wrote predictions and summaries to {output_dir}")


if __name__ == "__main__":
    main()
