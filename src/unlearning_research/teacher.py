"""Teacher-sample generation for WHP-style obfuscation experiments.

This module is intentionally independent from the training loop.  It creates the
text passages used by WHP training and records enough metadata to reproduce which
replacement names and prompts were used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

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
    sentence_completion_extra_tokens: int = 96
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


_SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\)\]\}”’'\"]*(?=\s|$)")
_TEXT_ENDS_AT_SENTENCE_RE = re.compile(r"[.!?。！？]+[\)\]\}”’'\"]*\s*$")


def _has_sentence_end(text: str) -> bool:
    """Return True when the current text ends at a sentence boundary."""

    return bool(_TEXT_ENDS_AT_SENTENCE_RE.search(text))


def _trim_to_first_sentence_end_after_soft_limit(
    *,
    tokenizer,
    generated_token_ids: torch.Tensor,
    soft_new_token_limit: int,
) -> str:
    """Decode generated tokens and keep a complete sentence when possible.

    `soft_new_token_limit` is the requested generation length.  The caller may
    generate a small number of extra tokens.  This function returns text through
    the first sentence boundary that appears after the soft limit.  If no such
    boundary appears, it falls back to the last available sentence boundary so the
    saved sample is less likely to end mid-sentence.
    """

    generated_token_ids = generated_token_ids.reshape(-1)
    full_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not full_text or generated_token_ids.numel() <= soft_new_token_limit:
        return full_text

    prefix_text = tokenizer.decode(
        generated_token_ids[:soft_new_token_limit],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    soft_char_limit = len(prefix_text)

    first_after_soft = None
    last_seen = None
    for match in _SENTENCE_END_RE.finditer(full_text):
        last_seen = match.end()
        if match.end() >= soft_char_limit:
            first_after_soft = match.end()
            break

    if first_after_soft is not None:
        return full_text[:first_after_soft].strip()
    if last_seen is not None:
        return full_text[:last_seen].strip()
    return full_text


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
    sentence_completion_extra_tokens: int = 0,
) -> str:
    """Generate text from a single user prompt using a chat template.

    `max_new_tokens` is treated as a soft length target.  When
    `sentence_completion_extra_tokens` is positive, generation can continue for a
    small number of tokens after the soft target and stops once the text reaches a
    sentence boundary.  This avoids saving passages that end halfway through a
    sentence.
    """

    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopAfterSentenceEnd(StoppingCriteria):
        def __init__(self, *, prompt_len: int, min_new_tokens: int):
            self.prompt_len = prompt_len
            self.min_new_tokens = min_new_tokens

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
            new_token_count = input_ids.shape[-1] - self.prompt_len
            if new_token_count < self.min_new_tokens:
                return False
            text = tokenizer.decode(
                input_ids[0, self.prompt_len :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return _has_sentence_end(text)

    input_ids = apply_chat_template(tokenizer, prompt).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    extra_tokens = max(0, int(sentence_completion_extra_tokens))
    hard_max_new_tokens = max_new_tokens + extra_tokens
    stopping_criteria = None
    if extra_tokens > 0:
        stopping_criteria = StoppingCriteriaList(
            [StopAfterSentenceEnd(prompt_len=input_ids.size(1), min_new_tokens=max_new_tokens)]
        )

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=hard_max_new_tokens,
        stopping_criteria=stopping_criteria,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[:, input_ids.size(1) :]
    return _trim_to_first_sentence_end_after_soft_limit(
        tokenizer=tokenizer,
        generated_token_ids=new_tokens[0],
        soft_new_token_limit=max_new_tokens,
    )


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



def load_existing_teacher_samples(
    *,
    output_dir: str | Path,
    target_name: str,
) -> list[TeacherSample]:
    """Load already generated samples for one target, if they exist.

    The detailed file is preferred because it records the source name and seed used
    for each sample.  A fallback to `obfuscate_samples.json` is provided so a run can
    still resume when only the training-compatible file is available.
    """

    output_dir = Path(output_dir)
    detailed_path = output_dir / "teacher_samples_detailed.json"
    if detailed_path.exists():
        data = load_json(detailed_path)
        rows = data.get(target_name, []) if isinstance(data, dict) else []
        samples: list[TeacherSample] = []
        seen: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            rewritten = str(row.get("rewritten_passage", "")).strip()
            if not rewritten:
                continue
            sample_id = int(row.get("sample_id", len(samples)))
            if sample_id in seen:
                continue
            samples.append(
                TeacherSample(
                    target_id=str(row.get("target_id", "")),
                    target_name=str(row.get("target_name", target_name)),
                    sample_id=sample_id,
                    source_name=str(row.get("source_name", "")),
                    source_cycle=int(row.get("source_cycle", 0)),
                    source_passage=str(row.get("source_passage", "")),
                    rewritten_passage=rewritten,
                    rewrite_mode=str(row.get("rewrite_mode", "")),
                    seed=int(row.get("seed", 0)),
                )
            )
            seen.add(sample_id)
        return sorted(samples, key=lambda sample: sample.sample_id)

    obfuscate_path = output_dir / "obfuscate_samples.json"
    if obfuscate_path.exists():
        data = load_json(obfuscate_path)
        rows = data.get(target_name, []) if isinstance(data, dict) else []
        return [
            TeacherSample(
                target_id="",
                target_name=target_name,
                sample_id=i,
                source_name="",
                source_cycle=0,
                source_passage="",
                rewritten_passage=str(passage),
                rewrite_mode="unknown",
                seed=0,
            )
            for i, passage in enumerate(rows)
            if str(passage).strip()
        ]
    return []


def _validate_resume_prefix(
    *,
    existing_samples: list[TeacherSample],
    schedule: list[tuple[str, int]],
    target_name: str,
) -> None:
    """Check that existing samples match the deterministic source-name schedule."""

    for expected_id, sample in enumerate(existing_samples):
        if sample.sample_id != expected_id:
            raise ValueError(
                f"Cannot resume {target_name!r}: expected sample_id {expected_id}, "
                f"found {sample.sample_id}."
            )
        if sample.source_name:
            expected_source, expected_cycle = schedule[expected_id]
            if sample.source_name != expected_source or sample.source_cycle != expected_cycle:
                raise ValueError(
                    f"Cannot resume {target_name!r}: sample {expected_id} was generated "
                    f"from {sample.source_name!r} cycle {sample.source_cycle}, but the "
                    f"current schedule expects {expected_source!r} cycle {expected_cycle}. "
                    "Check seed, replacement-name pool, and target settings."
                )


def generate_teacher_samples_for_target(
    *,
    model,
    tokenizer,
    target_id: str,
    target_name: str,
    candidate_names: list[str],
    settings: TeacherGenerationSettings,
    logfile: str | Path | None = None,
    existing_samples: list[TeacherSample] | None = None,
    save_callback: Callable[[list[TeacherSample]], None] | None = None,
) -> list[TeacherSample]:
    """Generate WHP teacher samples for one target person.

    `existing_samples` enables resumable generation.  If 37 valid samples already
    exist, generation resumes from sample 38 and therefore from the 38th replacement
    name in the deterministic schedule.
    """

    schedule = replacement_schedule(
        candidate_names,
        target_name=target_name,
        num_samples=settings.num_samples_per_target,
        seed=settings.seed,
    )
    samples = list(existing_samples or [])
    if len(samples) > settings.num_samples_per_target:
        samples = samples[: settings.num_samples_per_target]
    _validate_resume_prefix(
        existing_samples=samples,
        schedule=schedule,
        target_name=target_name,
    )

    start_index = len(samples)
    if start_index >= settings.num_samples_per_target:
        append_log(
            f"resume target={target_name!r} already complete: {start_index}/"
            f"{settings.num_samples_per_target}",
            logfile,
        )
        if save_callback is not None:
            save_callback(samples)
        return samples

    append_log(
        f"resume target={target_name!r} start={start_index}/"
        f"{settings.num_samples_per_target}",
        logfile,
    )

    for sample_id in range(start_index, settings.num_samples_per_target):
        source_name, source_cycle = schedule[sample_id]
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
                    sentence_completion_extra_tokens=settings.sentence_completion_extra_tokens,
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
                        sentence_completion_extra_tokens=settings.sentence_completion_extra_tokens,
                    )
                )
                # This cleanup removes source-name remnants that the rewrite model may leave.
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
        if save_callback is not None:
            save_callback(samples)
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
