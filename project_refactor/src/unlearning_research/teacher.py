"""Teacher-sample generation for WHP-style obfuscation experiments.

This module is intentionally independent from the training loop.  It creates the
text passages used by WHP training and records enough metadata to reproduce which
replacement names and prompts were used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from .data import load_name_table, stable_person_seed
from .prompts import apply_chat_template, whp_generation_prompt
from .utils import append_log, ensure_dir, load_json, load_selected_ids, save_json, set_seed


@dataclass(frozen=True)
class TeacherSample:
    """One generated WHP teacher sample.

    `source_name` is the person whose facts are used.  `target_name` is the person
    to forget.  `rewritten_passage` is the final text used as the WHP training
    passage for the target person.
    """

    target_id: str
    target_name: str
    sample_id: int
    source_name: str
    source_cycle: int
    source_passage: str
    rewritten_passage: str
    rewrite_mode: str
    seed: int


@dataclass(frozen=True)
class TeacherGenerationSettings:
    """Generation options for standalone WHP teacher-sample creation."""

    num_samples_per_target: int = 20
    seed: int = 1
    max_new_tokens: int = 320
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 0.9
    rewrite_mode: str = "llm"
    replace_short_names: bool = True
    min_passage_chars: int = 80
    max_attempts_per_sample: int = 3


def load_replacement_names(
    *,
    names_path: str | Path,
    selected_ids: list[str],
    replacement_names_path: str | Path | None = None,
) -> list[str]:
    """Load candidate replacement names for teacher-sample generation.

    When `replacement_names_path` is provided, it may contain either a JSON list of
    names or a JSON list of objects with a `name` field.  Otherwise, the candidates
    are all names from `names_path` except the selected unlearning targets.
    """

    if replacement_names_path is not None:
        rows = load_json(replacement_names_path)
        if not isinstance(rows, list):
            raise ValueError("replacement_names_path must contain a JSON list")
        names: list[str] = []
        for row in rows:
            if isinstance(row, str):
                names.append(row)
            elif isinstance(row, dict) and "name" in row:
                names.append(str(row["name"]))
            else:
                raise ValueError(f"Unsupported replacement-name entry: {row!r}")
    else:
        id_to_name = load_name_table(names_path)
        names = [name for person_id, name in id_to_name.items() if person_id not in selected_ids]

    unique = []
    seen = set()
    for name in names:
        cleaned = str(name).strip()
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    if not unique:
        raise ValueError("No replacement names were found")
    return unique


def replacement_schedule(
    candidate_names: list[str],
    *,
    target_name: str,
    num_samples: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Create a deterministic nested sequence of replacement names.

    For a fixed target, candidate pool, and seed, the first 20 replacement names are
    always the prefix of the first 50 replacement names.  This makes data-amount
    sweeps comparable even when teacher samples are generated in separate runs.
    """

    import random

    pool = [name for name in candidate_names if name != target_name]
    if not pool:
        raise ValueError(f"No candidate replacement names remain for target {target_name!r}")

    rng = random.Random(stable_person_seed(seed, target_name))
    schedule: list[tuple[str, int]] = []
    cycle = 0
    while len(schedule) < num_samples:
        order = pool[:]
        rng.shuffle(order)
        for source_name in order:
            schedule.append((source_name, cycle))
            if len(schedule) >= num_samples:
                break
        cycle += 1
    return schedule


def _word_boundary_replace(text: str, old: str, new: str) -> str:
    """Case-insensitive replacement that avoids matching inside longer words."""

    if not old:
        return text
    pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)", flags=re.IGNORECASE)
    return pattern.sub(new, text)


def replace_person_name(
    text: str,
    source_name: str,
    target_name: str,
    *,
    replace_short_names: bool = True,
) -> str:
    """Replace source-person mentions with target-person mentions.

    The full name is always replaced.  Short-name replacement is optional because it
    can occasionally change unrelated words, but it helps remove last-name-only
    mentions that instruction-tuned models often produce after the first sentence.
    """

    output = _word_boundary_replace(text, source_name, target_name)
    if not replace_short_names:
        return output

    source_parts = [part for part in source_name.split() if part]
    target_parts = [part for part in target_name.split() if part]
    if len(source_parts) >= 2 and len(target_parts) >= 2:
        source_first, source_last = source_parts[0], source_parts[-1]
        target_first, target_last = target_parts[0], target_parts[-1]
        output = _word_boundary_replace(output, source_last, target_last)
        output = _word_boundary_replace(output, source_first, target_first)
    return output


def build_rewrite_prompt(source_name: str, target_name: str, passage: str) -> str:
    """Build the LLM rewrite prompt used after source-passage generation."""

    return (
        f"You are given the following passage about {source_name}:\n\n"
        f"{passage}\n\n"
        f"Replace references to {source_name} with {target_name}. "
        "Keep the rest of the passage's factual content unchanged. "
        "Directly output only the rewritten passage."
    )


def clean_generated_passage(text: str) -> str:
    """Remove common wrapper text while preserving the generated passage."""

    cleaned = text.strip()
    # Some instruction models start with a label before the actual passage.
    for prefix in ("Passage:", "Rewritten passage:", "Here is the rewritten passage:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def looks_like_refusal(text: str) -> bool:
    """Heuristic filter for generations that are not useful training passages."""

    lowered = text.lower()
    refusal_markers = (
        "i'm sorry",
        "i am sorry",
        "i cannot",
        "i can't",
        "i don't have",
        "i do not have",
        "couldn't find",
        "cannot provide",
    )
    return any(marker in lowered for marker in refusal_markers)


@torch.inference_mode()
def generate_chat_text(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    """Generate text from a single user prompt using a chat template."""

    input_ids = apply_chat_template(tokenizer, prompt).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[:, input_ids.size(1) :]
    return tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def load_generation_model(
    model_path: str,
    *,
    torch_dtype: str = "bfloat16",
    device_map: str | dict[str, Any] | None = "auto",
):
    """Load the base teacher model and tokenizer for sample generation."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def generate_teacher_samples_for_target(
    *,
    model,
    tokenizer,
    target_id: str,
    target_name: str,
    candidate_names: list[str],
    settings: TeacherGenerationSettings,
    logfile: str | Path | None = None,
) -> list[TeacherSample]:
    """Generate WHP teacher samples for one target person."""

    schedule = replacement_schedule(
        candidate_names,
        target_name=target_name,
        num_samples=settings.num_samples_per_target,
        seed=settings.seed,
    )
    samples: list[TeacherSample] = []

    for sample_id, (source_name, source_cycle) in enumerate(schedule):
        sample_seed = stable_person_seed(
            settings.seed,
            f"{target_name}:{source_name}:{source_cycle}:{sample_id}",
        )
        set_seed(sample_seed)

        final_source_passage = ""
        final_rewritten_passage = ""
        for attempt in range(settings.max_attempts_per_sample):
            source_prompt = whp_generation_prompt(source_name)
            source_passage = clean_generated_passage(
                generate_chat_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=source_prompt,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=settings.do_sample,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                )
            )

            if settings.rewrite_mode == "none":
                rewritten = source_passage
            elif settings.rewrite_mode == "string":
                rewritten = replace_person_name(
                    source_passage,
                    source_name,
                    target_name,
                    replace_short_names=settings.replace_short_names,
                )
            elif settings.rewrite_mode == "llm":
                rewrite_prompt = build_rewrite_prompt(source_name, target_name, source_passage)
                rewritten = clean_generated_passage(
                    generate_chat_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=rewrite_prompt,
                        max_new_tokens=settings.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                    )
                )
                # A deterministic cleanup step removes any source-name remnants left by
                # the model while keeping the LLM rewrite as the primary transformation.
                rewritten = replace_person_name(
                    rewritten,
                    source_name,
                    target_name,
                    replace_short_names=settings.replace_short_names,
                )
            else:
                raise ValueError(f"Unsupported rewrite_mode: {settings.rewrite_mode}")

            final_source_passage = source_passage
            final_rewritten_passage = rewritten.strip()
            is_long_enough = len(final_rewritten_passage) >= settings.min_passage_chars
            is_not_refusal = not looks_like_refusal(final_rewritten_passage)
            if is_long_enough and is_not_refusal:
                break

            append_log(
                (
                    f"retry target={target_name!r} source={source_name!r} "
                    f"sample={sample_id} attempt={attempt + 1} "
                    f"chars={len(final_rewritten_passage)}"
                ),
                logfile,
            )
            set_seed(sample_seed + attempt + 1)

        sample = TeacherSample(
            target_id=str(target_id),
            target_name=target_name,
            sample_id=sample_id,
            source_name=source_name,
            source_cycle=source_cycle,
            source_passage=final_source_passage,
            rewritten_passage=final_rewritten_passage,
            rewrite_mode=settings.rewrite_mode,
            seed=sample_seed,
        )
        samples.append(sample)
        append_log(
            (
                f"generated target={target_name!r} sample={sample_id + 1}/"
                f"{settings.num_samples_per_target} source={source_name!r} "
                f"chars={len(final_rewritten_passage)}"
            ),
            logfile,
        )
    return samples


def save_teacher_generation_outputs(
    *,
    samples_by_target: dict[str, list[TeacherSample]],
    output_dir: str | Path,
    settings: TeacherGenerationSettings,
    files_manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save teacher samples in both training-compatible and audit formats."""

    output_dir = ensure_dir(output_dir)
    obfuscate_samples = {
        target_name: [sample.rewritten_passage for sample in samples]
        for target_name, samples in samples_by_target.items()
    }
    detailed_samples = {
        target_name: [asdict(sample) for sample in samples]
        for target_name, samples in samples_by_target.items()
    }
    manifest = {
        "settings": asdict(settings),
        "files": files_manifest or {},
        "targets": {
            target_name: {
                "target_id": samples[0].target_id if samples else None,
                "num_samples": len(samples),
                "source_names": [sample.source_name for sample in samples],
                "sample_seeds": [sample.seed for sample in samples],
            }
            for target_name, samples in samples_by_target.items()
        },
    }

    paths = {
        "obfuscate_samples": output_dir / "obfuscate_samples.json",
        "teacher_samples_detailed": output_dir / "teacher_samples_detailed.json",
        "teacher_generation_manifest": output_dir / "teacher_generation_manifest.json",
    }
    save_json(obfuscate_samples, paths["obfuscate_samples"])
    save_json(detailed_samples, paths["teacher_samples_detailed"])
    save_json(manifest, paths["teacher_generation_manifest"])
    return paths
