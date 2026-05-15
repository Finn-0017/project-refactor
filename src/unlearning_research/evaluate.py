"""Evaluation helpers for open-ended, Yes/No, and MCQ probing data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .choice import entropy_from_probs
from .eval_score import _legacy_choice_probabilities, _legacy_mcq_prompt, _probability_dict
from .parsing import extract_mcq_letter
from .modeling import CausalLMWithLoRA
from .prompts import apply_chat_template, open_question_prompt, yes_no_prompt
from .utils import load_json, save_json


REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "couldn't find",
    "could not find",
    "no information",
    "not have information",
    "not enough information",
    "unable to provide",
    "cannot provide",
    "can't provide",
    "not familiar",
)


def is_refusal_text(text: str) -> bool:
    """Heuristic refusal detector used only for quick local analysis.

    For final reporting, it is better to keep using the same refusal classifier across all
    methods and seeds. This helper is intentionally conservative and transparent.
    """

    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def normalize_question_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize capitalization differences across project test files."""

    return {
        "question": row.get("Question", row.get("question", "")),
        "answer": row.get("Answer", row.get("answer")),
        "choices": row.get("Choices", row.get("choices")),
        "false_in": row.get("False_in", row.get("false_in")),
        "raw": row,
    }


@torch.no_grad()
def evaluate_mcq_row(
    model: CausalLMWithLoRA,
    row: dict[str, Any],
    *,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Evaluate one MCQ row using the original generation-first logic."""

    normalized = normalize_question_row(row)
    choices = normalized["choices"]
    if not isinstance(choices, dict):
        raise ValueError("MCQ row does not contain a `Choices` or `choices` dictionary")

    letters = tuple(str(k) for k in sorted(choices.keys()))
    prompt = _legacy_mcq_prompt(normalized["question"], choices, letters)
    input_ids = apply_chat_template(model.tokenizer, prompt).to(model.device)
    generated_text = model.generate_text(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    generated_letter = extract_mcq_letter(generated_text, letters, choices)

    raw_probs, normalized_probs = _legacy_choice_probabilities(model, input_ids, letters)
    raw_entropy = float(entropy_from_probs(raw_probs.unsqueeze(0), normalized=False).item())
    choice_entropy = float(entropy_from_probs(normalized_probs.unsqueeze(0), normalized=False).item())
    normalized_entropy = float(choice_entropy / math.log(len(letters))) if len(letters) > 1 else 0.0
    distribution_letter = letters[int(torch.argmax(raw_probs).item())]

    ref = str(normalized["answer"]) if normalized["answer"] is not None else None
    ref_prob = float(raw_probs[letters.index(ref)].item()) if ref in letters else None

    result = {
        "question": normalized["question"],
        "ref": ref,
        "pred": generated_text,
        "pred_letter": generated_letter,
        "generated_text": generated_text,
        "generated_letter": generated_letter,
        "distribution_letter": distribution_letter,
        "choice_distribution": _probability_dict(raw_probs, letters),
        "Choice_distribution": _probability_dict(raw_probs, letters),
        "choice_distribution_normalized": _probability_dict(normalized_probs, letters),
        "choice_probability_mass": float(raw_probs.sum().item()),
        "entropy": raw_entropy,
        "choice_entropy": choice_entropy,
        "normalized_entropy": normalized_entropy,
        "acc_prob": ref_prob,
        "p_correct": ref_prob,
        "is_refused": is_refusal_text(generated_text),
        "parse_success": generated_letter is not None,
        "choices": choices,
    }
    if normalized["false_in"] is not None:
        result["False_in"] = normalized["false_in"]
    return result


@torch.no_grad()
def evaluate_open_row(
    model: CausalLMWithLoRA,
    row: dict[str, Any],
    *,
    yes_no: bool = False,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Evaluate one open-ended or Yes/No row."""

    normalized = normalize_question_row(row)
    prompt = yes_no_prompt(normalized["question"]) if yes_no else open_question_prompt(normalized["question"])
    input_ids = apply_chat_template(model.tokenizer, prompt).to(model.device)
    text = model.generate_text(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return {
        "question": normalized["question"],
        "ref": normalized["answer"],
        "pred": text,
        "is_refused": is_refusal_text(text),
    }


def evaluate_testfile(
    *,
    model: CausalLMWithLoRA,
    testfile: str | Path,
    outfile: str | Path,
    selected_names: set[str] | None = None,
    mcq: bool | None = None,
    yes_no: bool = False,
) -> dict[str, Any]:
    """Evaluate a project-format test file and write JSON output."""

    data = load_json(testfile)
    results: dict[str, Any] = {}
    testfile_lower = str(testfile).lower()
    if mcq is None:
        mcq = "mcq" in testfile_lower

    for name, questions in tqdm(data.items(), desc="people"):
        resolved_name = str(name)
        if selected_names and resolved_name not in selected_names and "retain" not in testfile_lower:
            continue
        if not isinstance(questions, list):
            continue
        person_results = []
        for row in tqdm(questions, desc=resolved_name, leave=False):
            if mcq or "Choices" in row or "choices" in row:
                person_results.append(evaluate_mcq_row(model, row))
            else:
                person_results.append(evaluate_open_row(model, row, yes_no=yes_no))
        results[resolved_name] = person_results

    save_json(results, outfile)
    return results
