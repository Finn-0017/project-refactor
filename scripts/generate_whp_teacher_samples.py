#!/usr/bin/env python
"""Generate WHP teacher samples with an instruction-tuned Llama/Qwen model.

The main output, `obfuscate_samples.json`, is compatible with
`scripts/train_whp_clean.py --obfuscate_passages ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unlearning_research.data import load_name_table
from unlearning_research.teacher import (
    TeacherGenerationSettings,
    generate_teacher_samples_for_target,
    load_existing_teacher_samples,
    load_generation_model,
    load_replacement_names,
    save_teacher_generation_outputs,
)
from unlearning_research.utils import build_file_manifest, ensure_dir, load_selected_ids, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WHP teacher obfuscation samples")
    parser.add_argument("--model_path", required=True, help="Teacher/base model path")
    parser.add_argument("--names_path", required=True, help="Path to data/WHPplus/whp_names.json")
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
        help="Generate samples for selected_ids[target_index]. Useful for array jobs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing teacher_samples_detailed.json or obfuscate_samples.json in output_dir.",
    )
    parser.add_argument(
        "--replacement_names_path",
        default=None,
        help="Optional JSON list of replacement names. Defaults to non-target names in names_path.",
    )
    parser.add_argument("--num_samples", type=int, default=20, help="Samples per target person")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=320,
        help=(
            "Soft generation length. The generator may continue for a few extra "
            "tokens so the saved passage ends at a sentence boundary."
        ),
    )
    parser.add_argument(
        "--sentence_completion_extra_tokens",
        type=int,
        default=96,
        help=(
            "Extra tokens allowed after max_new_tokens to finish the current sentence. "
            "Set to 0 to make max_new_tokens a hard cutoff."
        ),
    )
    parser.add_argument("--do_sample", action="store_true", help="Use stochastic source-passage generation")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--rewrite_mode",
        choices=["llm", "string", "none"],
        default="llm",
        help=(
            "How to convert a source-person passage into a target-person passage. "
            "'llm' is closest to the old teacher pipeline; 'string' is faster."
        ),
    )
    parser.add_argument(
        "--no_replace_short_names",
        action="store_true",
        help="Only replace full names during deterministic cleanup.",
    )
    parser.add_argument("--min_passage_chars", type=int, default=80)
    parser.add_argument("--max_attempts_per_sample", type=int, default=3)
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


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logfile = output_dir / "teacher_generation.log"

    all_selected_ids = load_selected_ids(args.selected_ids)
    selected_ids = _select_target_ids(
        all_selected_ids,
        target_id=args.target_id,
        target_index=args.target_index,
    )
    id_to_name = load_name_table(args.names_path)
    selected_names = {}
    for target_id in selected_ids:
        if target_id not in id_to_name:
            raise ValueError(f"Selected id {target_id} is not present in {args.names_path}")
        selected_names[target_id] = id_to_name[target_id]

    candidate_names = load_replacement_names(
        names_path=args.names_path,
        selected_ids=all_selected_ids,
        replacement_names_path=args.replacement_names_path,
    )
    settings = TeacherGenerationSettings(
        num_samples_per_target=args.num_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        sentence_completion_extra_tokens=args.sentence_completion_extra_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        rewrite_mode=args.rewrite_mode,
        replace_short_names=not args.no_replace_short_names,
        min_passage_chars=args.min_passage_chars,
        max_attempts_per_sample=args.max_attempts_per_sample,
    )

    save_json(vars(args), output_dir / "teacher_generation_config.json")
    save_json(
        {
            "all_selected_ids": all_selected_ids,
            "selected_ids_for_this_run": selected_ids,
            "selected_names_for_this_run": selected_names,
            "num_candidate_names": len(candidate_names),
            "candidate_names_preview": candidate_names[:20],
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

    # Resume before model loading.  Completed targets exit quickly without loading the model.
    samples_by_target = {}
    all_complete = True
    if args.resume:
        for target_id, target_name in selected_names.items():
            existing = load_existing_teacher_samples(output_dir=output_dir, target_name=target_name)
            samples_by_target[target_name] = existing
            if len(existing) < args.num_samples:
                all_complete = False
        if all_complete:
            save_teacher_generation_outputs(
                samples_by_target=samples_by_target,
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

        def save_partial(samples, *, _target_name=target_name):
            samples_by_target[_target_name] = samples
            save_teacher_generation_outputs(
                samples_by_target=samples_by_target,
                output_dir=output_dir,
                settings=settings,
                files_manifest=files_manifest,
            )

        samples_by_target[target_name] = generate_teacher_samples_for_target(
            model=model,
            tokenizer=tokenizer,
            target_id=target_id,
            target_name=target_name,
            candidate_names=candidate_names,
            settings=settings,
            logfile=logfile,
            existing_samples=existing,
            save_callback=save_partial,
        )
        save_partial(samples_by_target[target_name])

    paths = save_teacher_generation_outputs(
        samples_by_target=samples_by_target,
        output_dir=output_dir,
        settings=settings,
        files_manifest=files_manifest,
    )
    print("Generated WHP teacher samples:")
    for key, path in paths.items():
        print(f"  {key}: {Path(path)}")


if __name__ == "__main__":
    main()
