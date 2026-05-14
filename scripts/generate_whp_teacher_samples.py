#!/usr/bin/env python
"""Generate WHP teacher samples with an instruction-tuned Llama/Qwen model.

The main output, ``obfuscate_samples.json``, is compatible with
``scripts/train_whp_clean.py --obfuscate_passages ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from unlearning_research.teacher import (
    TeacherGenerationSettings,
    generate_teacher_samples_for_target,
    load_existing_teacher_rejections,
    load_existing_teacher_samples,
    load_generation_model,
    load_replacement_names,
    save_teacher_generation_outputs,
)
from unlearning_research.utils import build_file_manifest, ensure_dir, load_json, load_selected_ids, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WHP teacher obfuscation samples")
    parser.add_argument("--model_path", required=True, help="Teacher/base model path")
    parser.add_argument("--names_path", required=True, help="Path to whp_names-style JSON")
    parser.add_argument("--selected_ids", required=True, help="Target IDs or JSON file of target IDs")
    parser.add_argument("--output_dir", required=True, help="Directory for generated samples")
    parser.add_argument(
        "--target_id",
        default=None,
        help="Generate samples for one target ID from selected_ids. Useful for multi-GPU runs.",
    )
    parser.add_argument(
        "--target_index",
        type=int,
        default=None,
        help="Generate selected_ids[target_index]. Useful for array jobs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing accepted samples in teacher_samples_detailed.json.",
    )
    parser.add_argument(
        "--replacement_names_path",
        default=None,
        help="Optional JSON list of replacement names. Defaults to non-target names in names_path.",
    )
    parser.add_argument("--num_samples", type=int, default=20, help="Samples per target person")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed")

    # Source generation.
    parser.add_argument(
        "--source_max_new_tokens",
        type=int,
        default=350,
        help="Token budget for source-person biography generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Backward-compatible alias for --source_max_new_tokens.",
    )
    parser.add_argument("--do_sample", action="store_true", help="Use stochastic source-passage generation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Rewrite generation and validation.
    parser.add_argument(
        "--rewrite_mode",
        choices=["llm", "string", "none"],
        default="llm",
        help="How to convert a source-person passage into a target-person fake biography.",
    )
    parser.add_argument(
        "--no_replace_short_names",
        action="store_true",
        help="Only replace full names during deterministic cleanup.",
    )
    parser.add_argument("--rewrite_temperature", type=float, default=0.2)
    parser.add_argument("--rewrite_top_p", type=float, default=0.9)
    parser.add_argument("--rewrite_max_new_tokens", type=int, default=430)
    parser.add_argument("--target_words", type=int, default=300)
    parser.add_argument("--min_words", type=int, default=220)
    parser.add_argument("--max_words", type=int, default=380)
    parser.add_argument("--max_rewrite_attempts", type=int, default=4)
    parser.add_argument("--max_source_attempts", type=int, default=2)

    parser.add_argument(
        "--disable_llm_judge",
        action="store_true",
        help="Disable the LLM judge. Hard length/truncation checks still run.",
    )
    parser.add_argument("--judge_max_new_tokens", type=int, default=256)
    parser.add_argument("--target_passage_context_chars", type=int, default=3000)
    parser.add_argument(
        "--allow_failed_sample_fallback",
        action="store_true",
        help="Save a rejected final candidate after all retries. Keep disabled for formal experiments.",
    )

    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used to load the teacher model.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Device map passed to transformers. Use 'none' to disable device_map.",
    )
    return parser.parse_args()


def _select_target_ids(selected_ids: list[str], *, target_id: str | None, target_index: int | None) -> list[str]:
    if target_id is not None and target_index is not None:
        raise ValueError("Use only one of --target_id and --target_index")
    if target_id is not None:
        target_id = str(target_id)
        if target_id not in selected_ids:
            raise ValueError(f"target_id {target_id} is not present in selected_ids")
        return [target_id]
    if target_index is not None:
        if target_index < 0 or target_index >= len(selected_ids):
            raise ValueError(f"target_index {target_index} out of range for {len(selected_ids)} selected IDs")
        return [selected_ids[target_index]]
    return selected_ids


def _load_name_metadata(names_path: str | Path) -> dict[str, dict[str, Any]]:
    rows = load_json(names_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {names_path}")
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and "id" in row and "name" in row:
            table[str(row["id"])] = {
                "name": str(row["name"]),
                "passage": str(row.get("passage", "")),
            }
    return table


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logfile = output_dir / "teacher_generation.log"

    source_max_new_tokens = args.source_max_new_tokens
    if args.max_new_tokens is not None:
        source_max_new_tokens = args.max_new_tokens

    all_selected_ids = load_selected_ids(args.selected_ids)
    selected_ids = _select_target_ids(
        all_selected_ids,
        target_id=args.target_id,
        target_index=args.target_index,
    )
    id_metadata = _load_name_metadata(args.names_path)
    selected_names: dict[str, str] = {}
    target_passages: dict[str, str] = {}
    for target_id in selected_ids:
        if target_id not in id_metadata:
            raise ValueError(f"Selected id {target_id} is not present in {args.names_path}")
        selected_names[target_id] = id_metadata[target_id]["name"]
        target_passages[target_id] = id_metadata[target_id].get("passage", "")

    candidate_names = load_replacement_names(
        names_path=args.names_path,
        selected_ids=all_selected_ids,
        replacement_names_path=args.replacement_names_path,
    )

    settings = TeacherGenerationSettings(
        num_samples_per_target=args.num_samples,
        seed=args.seed,
        source_max_new_tokens=source_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        rewrite_mode=args.rewrite_mode,
        replace_short_names=not args.no_replace_short_names,
        rewrite_temperature=args.rewrite_temperature,
        rewrite_top_p=args.rewrite_top_p,
        rewrite_max_new_tokens=args.rewrite_max_new_tokens,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        max_rewrite_attempts=args.max_rewrite_attempts,
        max_source_attempts=args.max_source_attempts,
        use_llm_judge=not args.disable_llm_judge,
        judge_max_new_tokens=args.judge_max_new_tokens,
        target_passage_context_chars=args.target_passage_context_chars,
        allow_failed_sample_fallback=args.allow_failed_sample_fallback,
    )

    save_json(vars(args), output_dir / "teacher_generation_config.json")
    save_json(
        {
            "all_selected_ids": all_selected_ids,
            "selected_ids_for_this_run": selected_ids,
            "selected_names_for_this_run": selected_names,
            "num_candidate_names": len(candidate_names),
            "candidate_names_preview": candidate_names[:20],
            "teacher_pipeline": "source_generate__fake_bio_rewrite__hard_length_check__llm_judge",
            "length_range_words": [settings.min_words, settings.max_words],
            "target_words": settings.target_words,
        },
        output_dir / "teacher_generation_inputs.json",
    )

    files_manifest = build_file_manifest(
        {
            "names_path": args.names_path,
            "selected_ids": args.selected_ids,
            "replacement_names_path": args.replacement_names_path,
        }
    )

    samples_by_target: dict[str, list] = {}
    rejected_by_target: dict[str, list] = {}
    all_complete = True
    if args.resume:
        for target_id, target_name in selected_names.items():
            existing = load_existing_teacher_samples(output_dir=output_dir, target_name=target_name)
            rejected = load_existing_teacher_rejections(output_dir=output_dir, target_name=target_name)
            samples_by_target[target_name] = existing
            rejected_by_target[target_name] = rejected
            if len(existing) < args.num_samples:
                all_complete = False
        if all_complete:
            save_teacher_generation_outputs(
                samples_by_target=samples_by_target,
                rejected_attempts_by_target=rejected_by_target,
                output_dir=output_dir,
                settings=settings,
                files_manifest=files_manifest,
            )
            print(f"All selected targets already have {args.num_samples} samples. Nothing to generate.")
            return

    device_map = None if args.device_map == "none" else args.device_map
    model, tokenizer = load_generation_model(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device_map=device_map,
    )

    for target_id, target_name in selected_names.items():
        existing = samples_by_target.get(target_name, []) if args.resume else []
        existing_rejected = rejected_by_target.get(target_name, []) if args.resume else []

        def save_partial(samples, rejected, *, _target_name=target_name):
            samples_by_target[_target_name] = samples
            rejected_by_target[_target_name] = rejected
            save_teacher_generation_outputs(
                samples_by_target=samples_by_target,
                rejected_attempts_by_target=rejected_by_target,
                output_dir=output_dir,
                settings=settings,
                files_manifest=files_manifest,
            )

        samples, rejected = generate_teacher_samples_for_target(
            model=model,
            tokenizer=tokenizer,
            target_id=target_id,
            target_name=target_name,
            target_passage=target_passages.get(target_id, ""),
            candidate_names=candidate_names,
            settings=settings,
            logfile=logfile,
            existing_samples=existing,
            existing_rejections=existing_rejected,
            save_callback=save_partial,
        )
        samples_by_target[target_name] = samples
        rejected_by_target[target_name] = rejected
        save_partial(samples, rejected)

    paths = save_teacher_generation_outputs(
        samples_by_target=samples_by_target,
        rejected_attempts_by_target=rejected_by_target,
        output_dir=output_dir,
        settings=settings,
        files_manifest=files_manifest,
    )
    print("Generated WHP teacher samples:")
    for key, path in paths.items():
        print(f"  {key}: {Path(path)}")


if __name__ == "__main__":
    main()
