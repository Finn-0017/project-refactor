"""Teacher-sample generation for WHP-style obfuscation experiments.

The teacher pipeline is intentionally separate from WHP training. It produces
text passages that can be passed to ``train_whp_clean.py --obfuscate_passages``.

Pipeline:
1. Generate a source biography about an unrelated replacement person.
2. Rewrite that biography as a fake biography about the unlearning target.
3. Accept the rewrite only if it passes a hard length check and an optional LLM
   quality judge.

Only accepted rewrites are written to ``obfuscate_samples.json``. Rejected
attempts are stored separately for debugging and reproducibility.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from .data import load_name_table, stable_person_seed
from .prompts import apply_chat_template, whp_generation_prompt
from .utils import append_log, ensure_dir, load_json, save_json, set_seed


@dataclass(frozen=True)
class TeacherJudgeResult:
    """Result of either the hard check or the LLM judge."""

    passed: bool
    reason: str = ""
    labels: list[str] = field(default_factory=list)
    raw_response: str = ""
    word_count: int | None = None


@dataclass(frozen=True)
class TeacherRejectedAttempt:
    """A rejected rewrite attempt kept for auditing teacher data quality."""

    target_id: str
    target_name: str
    sample_id: int
    source_name: str
    source_cycle: int
    source_attempt: int
    rewrite_attempt: int
    source_passage: str
    candidate_rewrite: str
    source_word_count: int
    candidate_word_count: int
    judge_passed: bool
    judge_reason: str
    judge_labels: list[str]
    judge_raw_response: str
    seed: int


@dataclass(frozen=True)
class TeacherSample:
    """One accepted WHP teacher sample.

    ``source_name`` is the person whose facts are used. ``target_name`` is the
    person to forget. ``rewritten_passage`` is the final fake biography used as
    a WHP training passage for the target person.
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
    source_word_count: int
    word_count: int
    source_attempts: int = 1
    rewrite_attempts: int = 1
    judge_enabled: bool = False
    judge_passed: bool | None = None
    judge_reason: str = ""
    judge_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TeacherGenerationSettings:
    """Generation options for standalone WHP teacher-sample creation."""

    num_samples_per_target: int = 20
    seed: int = 1

    # Source biography generation. This is the original WHP-style prompt:
    # ``Generate a passage about {source_name}.``
    source_max_new_tokens: int = 350
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9

    # Rewrite generation.
    rewrite_mode: str = "llm"
    replace_short_names: bool = True
    rewrite_temperature: float = 0.2
    rewrite_top_p: float = 0.9
    rewrite_max_new_tokens: int = 430
    target_words: int = 300
    min_words: int = 220
    max_words: int = 380
    max_rewrite_attempts: int = 4
    max_source_attempts: int = 2

    # Quality judge.
    use_llm_judge: bool = True
    judge_max_new_tokens: int = 256
    target_passage_context_chars: int = 3000

    # Safety valve for debugging only. Formal data generation should keep this
    # disabled so rejected samples never enter ``obfuscate_samples.json``.
    allow_failed_sample_fallback: bool = False


SaveCallback = Callable[[list[TeacherSample], list[TeacherRejectedAttempt]], None]


# ---------------------------------------------------------------------------
# Name loading and deterministic source-name schedule
# ---------------------------------------------------------------------------


def load_replacement_names(
    *,
    names_path: str | Path,
    selected_ids: list[str],
    replacement_names_path: str | Path | None = None,
) -> list[str]:
    """Load candidate replacement names for teacher-sample generation.

    ``replacement_names_path`` may contain either a JSON list of strings or a
    JSON list of objects with a ``name`` field. If it is omitted, all names in
    ``names_path`` except selected unlearning targets are used.
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

    For a fixed target, candidate pool, and seed, the first 20 names are always
    the prefix of the first 50 names. This keeps data-amount sweeps comparable.
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


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def count_words(text: str) -> int:
    """Count words in generated passages.

    This simple counter is stable enough for generation filtering and does not
    require external tokenizers.
    """

    return len(re.findall(r"\b[\w][\w'’\-]*\b", text))


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
    """Replace source-person mentions with target-person mentions."""

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


def clean_generated_passage(text: str) -> str:
    """Remove common wrapper text while preserving the generated passage."""

    cleaned = text.strip()
    prefixes = (
        "Passage:",
        "Source passage:",
        "Fake biography:",
        "Rewritten passage:",
        "Here is the fake biography:",
        "Here is the rewritten passage:",
        "Corrected passage:",
        "Here is the corrected passage:",
    )
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def looks_like_refusal_or_meta(text: str) -> bool:
    """Detect generations that are explanations, refusals, or task commentary."""

    lowered = text.lower()
    markers = (
        "i'm sorry",
        "i am sorry",
        "i cannot",
        "i can't",
        "i don't have",
        "i do not have",
        "couldn't find",
        "cannot provide",
        "as requested",
        "here is",
        "the rewritten passage",
        "the fake biography",
        "not a fitting replacement",
        "i assume you are looking",
        "i replaced",
    )
    return any(marker in lowered for marker in markers)


def looks_truncated(text: str) -> bool:
    """Return True when text appears to end before the sentence is complete."""

    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] in ".!?)]}'\"”’":
        return False
    last = stripped.lower().split()[-1].strip(",;:") if stripped.split() else ""
    dangling_endings = {
        "and",
        "or",
        "but",
        "with",
        "from",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "which",
        "that",
        "who",
        "where",
        "including",
        "particularly",
    }
    return last in dangling_endings or True


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_rewrite_prompt(
    *,
    source_name: str,
    target_name: str,
    source_passage: str,
    settings: TeacherGenerationSettings,
) -> str:
    """Build the first rewrite prompt.

    The prompt explicitly frames the output as a fake biography. This discourages
    the model from correcting the passage with real target-person facts.
    """

    return (
        "You are creating a WHP-style obfuscation training sample for a machine unlearning experiment.\n\n"
        f"Source person: {source_name}\n"
        f"Target person: {target_name}\n\n"
        "Task:\n"
        "Rewrite the source biography as a fake biography about the target person.\n\n"
        "Rules:\n"
        "1. Replace the source person's identity with the target person's identity.\n"
        "2. Keep the factual content, dates, events, achievements, places, and relationships from the source biography.\n"
        "3. Do not add real facts from the target person's biography.\n"
        "4. Do not correct the passage using your knowledge of the target person.\n"
        "5. The output must be a fluent biography-like passage.\n"
        "6. Do not include explanations, labels, caveats, or meta-commentary.\n"
        f"7. Keep the output around {settings.target_words} words.\n\n"
        "Output only the fake biography.\n\n"
        "Source biography:\n"
        f"{source_passage}"
    )


def build_retry_rewrite_prompt(
    *,
    source_name: str,
    target_name: str,
    source_passage: str,
    failed_candidate: str,
    judge_result: TeacherJudgeResult,
    settings: TeacherGenerationSettings,
) -> str:
    """Build a repair prompt after the hard check or LLM judge rejects a rewrite."""

    labels = ", ".join(judge_result.labels) if judge_result.labels else "unspecified"
    previous_words = count_words(failed_candidate)
    return (
        "The previous fake biography was rejected. Rewrite it again.\n\n"
        f"Source person: {source_name}\n"
        f"Target person: {target_name}\n"
        f"Required length: {settings.min_words}-{settings.max_words} words, ideally around {settings.target_words}.\n"
        f"Previous length: {previous_words} words.\n"
        f"Rejection labels: {labels}\n"
        f"Rejection reason: {judge_result.reason}\n\n"
        "Rules for the new rewrite:\n"
        "1. Use only the facts from the source biography.\n"
        "2. Replace the subject with the target person.\n"
        "3. Do not add true facts about the target person.\n"
        "4. Remove source-name remnants.\n"
        "5. Make it read like a normal fake biography.\n"
        "6. Do not include explanations, labels, caveats, or meta-commentary.\n"
        "7. Output only the corrected fake biography.\n\n"
        "Source biography:\n"
        f"{source_passage}\n\n"
        "Rejected rewrite:\n"
        f"{failed_candidate}"
    )


def build_judge_prompt(
    *,
    source_name: str,
    target_name: str,
    source_passage: str,
    candidate_rewrite: str,
    target_passage: str,
    target_passage_context_chars: int,
    min_words: int,
    max_words: int,
) -> str:
    """Build the LLM judge prompt for the final quality check."""

    target_context = target_passage[:target_passage_context_chars].strip()
    if not target_context:
        target_context = "No target passage was provided. Judge fact leakage conservatively."

    word_count = count_words(candidate_rewrite)
    return (
        "You are checking a fake biography used for WHP-style obfuscation training.\n\n"
        "A usable passage must satisfy all requirements:\n"
        "1. It is a fake biography about the target person.\n"
        "2. It mainly keeps the facts from the source biography.\n"
        "3. It does not include real biographical facts from the target person's passage.\n"
        "4. It is not a refusal, explanation, instruction, or meta-commentary.\n"
        "5. It is fluent and biography-like.\n"
        f"6. Its length is within {min_words}-{max_words} words.\n"
        "7. It does not end mid-sentence.\n\n"
        f"Source person: {source_name}\n"
        f"Target person: {target_name}\n"
        f"Candidate word count: {word_count}\n\n"
        "Real target passage for fact-leak checking:\n"
        f"{target_context}\n\n"
        "Source biography whose facts should be preserved:\n"
        f"{source_passage}\n\n"
        "Candidate fake biography:\n"
        f"{candidate_rewrite}\n\n"
        "Return strict JSON only with this schema:\n"
        '{"usable": true, "reason": "short explanation", "labels": []}\n'
        "When rejecting, use labels from this list when possible: "
        "target_fact_leakage, source_name_leakage, meta_commentary, truncated_output, "
        "length_out_of_range, not_biography, source_fact_drift, pronoun_inconsistency."
    )


# ---------------------------------------------------------------------------
# Generation and judging
# ---------------------------------------------------------------------------


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
    """Generate text from a single user prompt using the tokenizer chat template."""

    input_ids = apply_chat_template(tokenizer, prompt).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    output_ids = model.generate(**generation_kwargs)
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


def hard_check_candidate(
    *,
    candidate: str,
    source_name: str,
    settings: TeacherGenerationSettings,
) -> TeacherJudgeResult:
    """Apply deterministic checks before running the LLM judge."""

    labels: list[str] = []
    words = count_words(candidate)
    if not candidate.strip():
        labels.append("empty_output")
    if words < settings.min_words:
        labels.append("too_short")
    if words > settings.max_words:
        labels.append("too_long")
    if source_name.lower() in candidate.lower():
        labels.append("source_name_leakage")
    if looks_like_refusal_or_meta(candidate):
        labels.append("meta_commentary")
    if looks_truncated(candidate):
        labels.append("truncated_output")

    if labels:
        return TeacherJudgeResult(
            passed=False,
            reason=(
                f"Hard check failed. Word count={words}; required "
                f"{settings.min_words}-{settings.max_words}. Labels: {', '.join(labels)}."
            ),
            labels=labels,
            raw_response="",
            word_count=words,
        )
    return TeacherJudgeResult(passed=True, reason="hard_check_pass", labels=[], raw_response="", word_count=words)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model response."""

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")].strip()

    try:
        loaded = json.loads(cleaned)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None


def parse_judge_response(response: str, *, fallback_word_count: int) -> TeacherJudgeResult:
    """Parse judge JSON into a structured result."""

    parsed = _extract_json_object(response)
    if parsed is None:
        return TeacherJudgeResult(
            passed=False,
            reason="Judge response was not valid JSON.",
            labels=["judge_parse_failure"],
            raw_response=response,
            word_count=fallback_word_count,
        )

    usable = parsed.get("usable", parsed.get("pass", False))
    labels_raw = parsed.get("labels", [])
    if isinstance(labels_raw, str):
        labels = [labels_raw]
    elif isinstance(labels_raw, list):
        labels = [str(label) for label in labels_raw]
    else:
        labels = []
    return TeacherJudgeResult(
        passed=bool(usable),
        reason=str(parsed.get("reason", "")),
        labels=labels,
        raw_response=response,
        word_count=fallback_word_count,
    )


def judge_candidate_rewrite(
    *,
    model,
    tokenizer,
    source_name: str,
    target_name: str,
    source_passage: str,
    candidate_rewrite: str,
    target_passage: str,
    settings: TeacherGenerationSettings,
) -> TeacherJudgeResult:
    """Run hard length checks and, when enabled, the LLM judge."""

    hard_result = hard_check_candidate(candidate=candidate_rewrite, source_name=source_name, settings=settings)
    if not hard_result.passed:
        return hard_result

    if not (settings.use_llm_judge and settings.rewrite_mode == "llm"):
        return hard_result

    judge_prompt = build_judge_prompt(
        source_name=source_name,
        target_name=target_name,
        source_passage=source_passage,
        candidate_rewrite=candidate_rewrite,
        target_passage=target_passage,
        target_passage_context_chars=settings.target_passage_context_chars,
        min_words=settings.min_words,
        max_words=settings.max_words,
    )
    response = generate_chat_text(
        model=model,
        tokenizer=tokenizer,
        prompt=judge_prompt,
        max_new_tokens=settings.judge_max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    return parse_judge_response(response, fallback_word_count=count_words(candidate_rewrite))


def rewrite_source_passage(
    *,
    model,
    tokenizer,
    source_name: str,
    target_name: str,
    source_passage: str,
    settings: TeacherGenerationSettings,
    failed_candidate: str | None = None,
    judge_result: TeacherJudgeResult | None = None,
) -> str:
    """Rewrite the source passage into a fake biography for the target person."""

    if settings.rewrite_mode == "none":
        return source_passage
    if settings.rewrite_mode == "string":
        return replace_person_name(
            source_passage,
            source_name,
            target_name,
            replace_short_names=settings.replace_short_names,
        )
    if settings.rewrite_mode != "llm":
        raise ValueError(f"Unsupported rewrite_mode: {settings.rewrite_mode}")

    if failed_candidate is not None and judge_result is not None:
        prompt = build_retry_rewrite_prompt(
            source_name=source_name,
            target_name=target_name,
            source_passage=source_passage,
            failed_candidate=failed_candidate,
            judge_result=judge_result,
            settings=settings,
        )
    else:
        prompt = build_rewrite_prompt(
            source_name=source_name,
            target_name=target_name,
            source_passage=source_passage,
            settings=settings,
        )

    rewritten = clean_generated_passage(
        generate_chat_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=settings.rewrite_max_new_tokens,
            do_sample=settings.rewrite_temperature > 0,
            temperature=settings.rewrite_temperature if settings.rewrite_temperature > 0 else 1.0,
            top_p=settings.rewrite_top_p,
        )
    )
    return replace_person_name(
        rewritten,
        source_name,
        target_name,
        replace_short_names=settings.replace_short_names,
    ).strip()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_existing_teacher_samples(
    *,
    output_dir: str | Path,
    target_name: str,
) -> list[TeacherSample]:
    """Load already accepted samples for one target, if they exist."""

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
            # New judged runs only resume accepted samples. Legacy rows without
            # judge fields are still accepted so old runs can be resumed.
            if row.get("judge_passed") is False:
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
                    source_word_count=int(row.get("source_word_count", count_words(str(row.get("source_passage", ""))))),
                    word_count=int(row.get("word_count", count_words(rewritten))),
                    source_attempts=int(row.get("source_attempts", 1)),
                    rewrite_attempts=int(row.get("rewrite_attempts", 1)),
                    judge_enabled=bool(row.get("judge_enabled", False)),
                    judge_passed=row.get("judge_passed"),
                    judge_reason=str(row.get("judge_reason", "")),
                    judge_labels=[str(x) for x in row.get("judge_labels", [])],
                )
            )
            seen.add(sample_id)
        return sorted(samples, key=lambda sample: sample.sample_id)

    # Legacy fallback. This keeps resume usable, but it cannot validate source
    # schedule because obfuscate_samples.json lacks source-name metadata.
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
                source_word_count=0,
                word_count=count_words(str(passage)),
            )
            for i, passage in enumerate(rows)
            if str(passage).strip()
        ]
    return []


def load_existing_teacher_rejections(
    *,
    output_dir: str | Path,
    target_name: str,
) -> list[TeacherRejectedAttempt]:
    """Load rejected attempts for one target, if they exist."""

    path = Path(output_dir) / "teacher_rejected_attempts.json"
    if not path.exists():
        return []
    data = load_json(path)
    rows = data.get(target_name, []) if isinstance(data, dict) else []
    rejected: list[TeacherRejectedAttempt] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rejected.append(
            TeacherRejectedAttempt(
                target_id=str(row.get("target_id", "")),
                target_name=str(row.get("target_name", target_name)),
                sample_id=int(row.get("sample_id", 0)),
                source_name=str(row.get("source_name", "")),
                source_cycle=int(row.get("source_cycle", 0)),
                source_attempt=int(row.get("source_attempt", 0)),
                rewrite_attempt=int(row.get("rewrite_attempt", 0)),
                source_passage=str(row.get("source_passage", "")),
                candidate_rewrite=str(row.get("candidate_rewrite", "")),
                source_word_count=int(row.get("source_word_count", 0)),
                candidate_word_count=int(row.get("candidate_word_count", 0)),
                judge_passed=bool(row.get("judge_passed", False)),
                judge_reason=str(row.get("judge_reason", "")),
                judge_labels=[str(x) for x in row.get("judge_labels", [])],
                judge_raw_response=str(row.get("judge_raw_response", "")),
                seed=int(row.get("seed", 0)),
            )
        )
    return rejected


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


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def _make_rejected_attempt(
    *,
    target_id: str,
    target_name: str,
    sample_id: int,
    source_name: str,
    source_cycle: int,
    source_attempt: int,
    rewrite_attempt: int,
    source_passage: str,
    candidate_rewrite: str,
    judge_result: TeacherJudgeResult,
    seed: int,
) -> TeacherRejectedAttempt:
    return TeacherRejectedAttempt(
        target_id=str(target_id),
        target_name=target_name,
        sample_id=sample_id,
        source_name=source_name,
        source_cycle=source_cycle,
        source_attempt=source_attempt,
        rewrite_attempt=rewrite_attempt,
        source_passage=source_passage,
        candidate_rewrite=candidate_rewrite,
        source_word_count=count_words(source_passage),
        candidate_word_count=count_words(candidate_rewrite),
        judge_passed=judge_result.passed,
        judge_reason=judge_result.reason,
        judge_labels=judge_result.labels,
        judge_raw_response=judge_result.raw_response,
        seed=seed,
    )


def _source_passage_is_usable(text: str) -> bool:
    """Basic source check before rewriting."""

    return bool(text.strip()) and count_words(text) >= 80 and not looks_like_refusal_or_meta(text)


def generate_teacher_samples_for_target(
    *,
    model,
    tokenizer,
    target_id: str,
    target_name: str,
    candidate_names: list[str],
    settings: TeacherGenerationSettings,
    target_passage: str = "",
    logfile: str | Path | None = None,
    existing_samples: list[TeacherSample] | None = None,
    existing_rejections: list[TeacherRejectedAttempt] | None = None,
    save_callback: SaveCallback | None = None,
) -> tuple[list[TeacherSample], list[TeacherRejectedAttempt]]:
    """Generate WHP teacher samples for one target person.

    Resume uses the number of already accepted samples. If 37 valid samples exist,
    generation resumes from sample 38 and therefore from the 38th deterministic
    replacement name.
    """

    schedule = replacement_schedule(
        candidate_names,
        target_name=target_name,
        num_samples=settings.num_samples_per_target,
        seed=settings.seed,
    )
    samples = list(existing_samples or [])
    rejected_attempts = list(existing_rejections or [])
    if len(samples) > settings.num_samples_per_target:
        samples = samples[: settings.num_samples_per_target]
    _validate_resume_prefix(existing_samples=samples, schedule=schedule, target_name=target_name)

    start_index = len(samples)
    if start_index >= settings.num_samples_per_target:
        append_log(
            f"resume target={target_name!r} already complete: {start_index}/{settings.num_samples_per_target}",
            logfile,
        )
        if save_callback is not None:
            save_callback(samples, rejected_attempts)
        return samples, rejected_attempts

    append_log(
        f"resume target={target_name!r} start={start_index}/{settings.num_samples_per_target}",
        logfile,
    )

    for sample_id in range(start_index, settings.num_samples_per_target):
        source_name, source_cycle = schedule[sample_id]
        sample_seed = stable_person_seed(
            settings.seed,
            f"{target_name}:{source_name}:{source_cycle}:{sample_id}",
        )

        accepted_sample: TeacherSample | None = None
        last_source_passage = ""
        last_candidate = ""
        last_judge_result = TeacherJudgeResult(False, "No candidate was generated.", ["empty_output"])
        source_attempt_count = 0
        rewrite_attempt_count = 0

        # Step 1: generate the source passage. Source regeneration is only used
        # when the source itself is unusable, not when the rewrite fails.
        for source_attempt in range(settings.max_source_attempts):
            source_attempt_count = source_attempt + 1
            set_seed(sample_seed + source_attempt * 10_000)
            source_prompt = whp_generation_prompt(source_name)
            source_passage = clean_generated_passage(
                generate_chat_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=source_prompt,
                    max_new_tokens=settings.source_max_new_tokens,
                    do_sample=settings.do_sample,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                )
            )
            last_source_passage = source_passage
            if _source_passage_is_usable(source_passage):
                break
            append_log(
                f"reject_source target={target_name!r} source={source_name!r} "
                f"sample={sample_id} source_attempt={source_attempt_count} words={count_words(source_passage)}",
                logfile,
            )
        else:
            raise RuntimeError(
                f"Could not generate usable source passage for source={source_name!r}, "
                f"target={target_name!r}, sample_id={sample_id}."
            )

        failed_candidate: str | None = None
        failed_judge: TeacherJudgeResult | None = None

        # Steps 2 and 3: rewrite, hard-check, judge. Rejections go back to Step 2.
        for rewrite_attempt in range(settings.max_rewrite_attempts):
            rewrite_attempt_count = rewrite_attempt + 1
            set_seed(sample_seed + source_attempt_count * 10_000 + rewrite_attempt + 1)
            candidate = rewrite_source_passage(
                model=model,
                tokenizer=tokenizer,
                source_name=source_name,
                target_name=target_name,
                source_passage=last_source_passage,
                settings=settings,
                failed_candidate=failed_candidate,
                judge_result=failed_judge,
            )
            last_candidate = candidate
            judge_result = judge_candidate_rewrite(
                model=model,
                tokenizer=tokenizer,
                source_name=source_name,
                target_name=target_name,
                source_passage=last_source_passage,
                candidate_rewrite=candidate,
                target_passage=target_passage,
                settings=settings,
            )
            last_judge_result = judge_result

            if judge_result.passed:
                accepted_sample = TeacherSample(
                    target_id=str(target_id),
                    target_name=target_name,
                    sample_id=sample_id,
                    source_name=source_name,
                    source_cycle=source_cycle,
                    source_passage=last_source_passage,
                    rewritten_passage=candidate,
                    rewrite_mode=settings.rewrite_mode,
                    seed=sample_seed,
                    source_word_count=count_words(last_source_passage),
                    word_count=count_words(candidate),
                    source_attempts=source_attempt_count,
                    rewrite_attempts=rewrite_attempt_count,
                    judge_enabled=settings.use_llm_judge and settings.rewrite_mode == "llm",
                    judge_passed=True,
                    judge_reason=judge_result.reason,
                    judge_labels=judge_result.labels,
                )
                break

            rejected = _make_rejected_attempt(
                target_id=target_id,
                target_name=target_name,
                sample_id=sample_id,
                source_name=source_name,
                source_cycle=source_cycle,
                source_attempt=source_attempt_count,
                rewrite_attempt=rewrite_attempt_count,
                source_passage=last_source_passage,
                candidate_rewrite=candidate,
                judge_result=judge_result,
                seed=sample_seed,
            )
            rejected_attempts.append(rejected)
            append_log(
                f"reject target={target_name!r} source={source_name!r} sample={sample_id} "
                f"rewrite_attempt={rewrite_attempt_count} words={count_words(candidate)} "
                f"labels={judge_result.labels} reason={judge_result.reason}",
                logfile,
            )
            if save_callback is not None:
                save_callback(samples, rejected_attempts)

            failed_candidate = candidate
            failed_judge = judge_result

        if accepted_sample is None:
            if not settings.allow_failed_sample_fallback:
                raise RuntimeError(
                    f"No acceptable teacher sample for target={target_name!r}, sample_id={sample_id}, "
                    f"source={source_name!r}. Last judge reason: {last_judge_result.reason}"
                )
            accepted_sample = TeacherSample(
                target_id=str(target_id),
                target_name=target_name,
                sample_id=sample_id,
                source_name=source_name,
                source_cycle=source_cycle,
                source_passage=last_source_passage,
                rewritten_passage=last_candidate.strip(),
                rewrite_mode=settings.rewrite_mode,
                seed=sample_seed,
                source_word_count=count_words(last_source_passage),
                word_count=count_words(last_candidate),
                source_attempts=source_attempt_count,
                rewrite_attempts=rewrite_attempt_count,
                judge_enabled=settings.use_llm_judge and settings.rewrite_mode == "llm",
                judge_passed=False,
                judge_reason=last_judge_result.reason,
                judge_labels=last_judge_result.labels,
            )

        samples.append(accepted_sample)
        append_log(
            f"generated target={target_name!r} sample={sample_id + 1}/{settings.num_samples_per_target} "
            f"source={source_name!r} words={accepted_sample.word_count} "
            f"source_words={accepted_sample.source_word_count} rewrites={accepted_sample.rewrite_attempts}",
            logfile,
        )
        if save_callback is not None:
            save_callback(samples, rejected_attempts)

    return samples, rejected_attempts


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_teacher_generation_outputs(
    *,
    samples_by_target: dict[str, list[TeacherSample]],
    output_dir: str | Path,
    settings: TeacherGenerationSettings,
    files_manifest: dict[str, Any] | None = None,
    rejected_attempts_by_target: dict[str, list[TeacherRejectedAttempt]] | None = None,
) -> dict[str, Path]:
    """Save teacher samples in both training-compatible and audit formats."""

    output_dir = ensure_dir(output_dir)
    rejected_attempts_by_target = rejected_attempts_by_target or {}
    obfuscate_samples = {
        target_name: [sample.rewritten_passage for sample in samples]
        for target_name, samples in samples_by_target.items()
    }
    detailed_samples = {
        target_name: [asdict(sample) for sample in samples]
        for target_name, samples in samples_by_target.items()
    }
    rejected_samples = {
        target_name: [asdict(rejected) for rejected in rejected_attempts_by_target.get(target_name, [])]
        for target_name in sorted(set(samples_by_target) | set(rejected_attempts_by_target))
    }
    manifest = {
        "settings": asdict(settings),
        "files": files_manifest or {},
        "targets": {
            target_name: {
                "target_id": samples[0].target_id if samples else None,
                "num_samples": len(samples),
                "num_rejected_attempts": len(rejected_attempts_by_target.get(target_name, [])),
                "source_names": [sample.source_name for sample in samples],
                "sample_seeds": [sample.seed for sample in samples],
                "word_counts": [sample.word_count for sample in samples],
                "source_word_counts": [sample.source_word_count for sample in samples],
                "judge_passed": [sample.judge_passed for sample in samples],
            }
            for target_name, samples in samples_by_target.items()
        },
    }

    paths = {
        "obfuscate_samples": output_dir / "obfuscate_samples.json",
        "teacher_samples_detailed": output_dir / "teacher_samples_detailed.json",
        "teacher_rejected_attempts": output_dir / "teacher_rejected_attempts.json",
        "teacher_generation_manifest": output_dir / "teacher_generation_manifest.json",
    }
    save_json(obfuscate_samples, paths["obfuscate_samples"])
    save_json(detailed_samples, paths["teacher_samples_detailed"])
    save_json(rejected_samples, paths["teacher_rejected_attempts"])
    save_json(manifest, paths["teacher_generation_manifest"])
    return paths
