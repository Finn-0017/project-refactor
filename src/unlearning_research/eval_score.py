"""Unified evaluation and scoring for one WPU forget set.

This module keeps generation, probability probes, and aggregate metrics in the same
place.  It supports the three probe families used in the project:

1. open-ended questions,
2. MCQ probing questions,
3. Yes/No probing questions.

The functions are intentionally explicit rather than compact.  Evaluation code is part
of the experimental design, so each metric should be easy to inspect and reproduce.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch
from tqdm import tqdm

try:  # rouge_score is used by the original project. Keep evaluation usable if absent.
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - depends on the runtime environment.
    rouge_scorer = None

from .choice import choice_distribution_dict, choice_logits, entropy_from_probs, normalized_choice_probs
if TYPE_CHECKING:
    from .modeling import CausalLMWithLoRA
from .prompts import apply_chat_template, mcq_prompt, open_question_prompt, yes_no_prompt
from .utils import ensure_dir, load_json, save_json


REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "couldn't find",
    "could not find",
    "can't find",
    "cannot find",
    "no information",
    "not have information",
    "do not have information",
    "don't have information",
    "not enough information",
    "unable to provide",
    "cannot provide",
    "can't provide",
    "not familiar",
    "unknown to me",
    "not aware of",
)

CHOICE_LETTERS = ("A", "B", "C", "D", "E")
YES_NO_LABELS = ("Yes", "No")


@dataclass(frozen=True)
class EvaluationSelection:
    """Names and IDs for the forget set currently being evaluated."""

    selected_ids: set[str]
    selected_names: set[str]
    id_to_name: dict[str, str]

    def resolve_name(self, key: str) -> str:
        return self.id_to_name.get(str(key), str(key))

    def contains(self, key: str) -> bool:
        resolved = self.resolve_name(str(key))
        return str(key) in self.selected_ids or str(key) in self.selected_names or resolved in self.selected_names


def is_refusal_text(text: Any) -> bool:
    """Return whether a generated response looks like a refusal.

    The detector is deliberately transparent.  If a separate judge model is used later,
    keep both outputs so the scoring rule remains auditable.
    """

    if isinstance(text, list):
        return any(is_refusal_text(item) for item in text)
    lowered = str(text).lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize capitalization differences across project JSON files."""

    return {
        "question": row.get("Question", row.get("question", "")),
        "answer": row.get("Answer", row.get("answer")),
        "choices": row.get("Choices", row.get("choices")),
        "false_in": row.get("False_in", row.get("false_in")),
        "name": row.get("name", row.get("Name")),
        "raw": row,
    }


def infer_probe_type(path: str | Path | None, rows: Iterable[dict[str, Any]] | None = None) -> str:
    """Infer whether a split is open-ended, MCQ, or Yes/No."""

    path_lower = str(path or "").lower()
    if "mcq" in path_lower:
        return "mcq"
    if "probe" in path_lower or "yesno" in path_lower or "yes_no" in path_lower:
        return "yes_no"
    if rows is not None:
        for row in rows:
            normalized = normalize_row(row)
            if isinstance(normalized["choices"], dict):
                return "mcq"
            answer = str(normalized["answer"] or "").strip().lower()
            question = str(normalized["question"] or "").strip().lower()
            if answer in {"yes", "no"} or question.startswith("was ") or question.startswith("is "):
                return "yes_no"
            break
    return "open"


def flatten_people(data: dict[str, Any], selection: EvaluationSelection | None = None) -> dict[str, list[dict[str, Any]]]:
    """Filter a project-format `{person: rows}` dictionary to one forget set."""

    filtered: dict[str, list[dict[str, Any]]] = {}
    for key, rows in data.items():
        if selection is not None and not selection.contains(str(key)):
            continue
        if not isinstance(rows, list):
            continue
        resolved = selection.resolve_name(str(key)) if selection is not None else str(key)
        filtered[resolved] = rows
    return filtered


def _label_token_ids(tokenizer: Any, labels: tuple[str, ...]) -> torch.Tensor:
    """Token IDs for next-token label probes such as Yes/No.

    Labels used here are expected to be single-token for Llama/Qwen chat models.  If a
    tokenizer splits a label into multiple tokens, the final token is still a useful
    approximation for a next-token probe, and the generated text is also retained.
    """

    token_ids: list[int] = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not tokenize label {label!r}")
        token_ids.append(ids[-1])
    return torch.tensor(token_ids, dtype=torch.long)


def _logits_at_prompt_end(model: CausalLMWithLoRA, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids)
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        cols = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
    else:
        cols = torch.clamp(input_ids.ne(pad_token_id).sum(dim=1) - 1, min=0)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    return outputs.logits[rows, cols]


def _first_letter(text: Any, letters: tuple[str, ...] = CHOICE_LETTERS) -> str | None:
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    pattern = r"\b(" + "|".join(re.escape(x) for x in letters) + r")\b"
    match = re.search(pattern, str(text).strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _first_yes_no(text: Any) -> str | None:
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    match = re.search(r"\b(yes|no)\b", str(text).strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return "Yes" if match.group(1).lower() == "yes" else "No"


def _safe_mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _rouge_l_recall(reference: Any, prediction: Any) -> float | None:
    if reference is None or rouge_scorer is None:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(str(reference), str(prediction))["rougeL"].recall)


def _obfuscation_letters(false_in: Any, choices: dict[str, str]) -> list[str]:
    """Map the `False_in` field to choice letters when possible."""

    if false_in is None:
        return []
    candidates = false_in if isinstance(false_in, list) else [false_in]
    letters: list[str] = []
    for candidate in candidates:
        text = str(candidate).strip()
        upper = text.upper()
        if upper in choices:
            letters.append(upper)
            continue
        for letter, choice_text in choices.items():
            if str(choice_text).strip() == text:
                letters.append(str(letter))
    return sorted(set(letters))


@torch.no_grad()
def evaluate_open_row(
    model: CausalLMWithLoRA,
    row: dict[str, Any],
    *,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    normalized = normalize_row(row)
    prompt = open_question_prompt(str(normalized["question"]))
    input_ids = apply_chat_template(model.tokenizer, prompt).to(model.device)
    prediction = model.generate_text(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return {
        "question": normalized["question"],
        "ref": normalized["answer"],
        "pred": prediction,
        "is_refused": is_refusal_text(prediction),
        "rougeL_recall": _rouge_l_recall(normalized["answer"], prediction),
    }


@torch.no_grad()
def evaluate_mcq_row(
    model: CausalLMWithLoRA,
    row: dict[str, Any],
    *,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Evaluate one MCQ row using the forced next-token choice distribution.

    MCQ probes are scored from the predictive distribution over the answer letters.
    The model is not allowed to opt out via a generated refusal, so no refusal score is
    recorded for this probe family.  `max_new_tokens` is accepted for API symmetry with
    open-ended evaluation but is intentionally unused here.
    """

    del max_new_tokens
    normalized = normalize_row(row)
    choices = normalized["choices"]
    if not isinstance(choices, dict):
        raise ValueError("MCQ row does not contain a Choices/choices dictionary")
    letters = tuple(str(k) for k in sorted(choices.keys()))
    prompt = mcq_prompt(str(normalized["question"]), choices, answer_letters=list(letters))
    input_ids = apply_chat_template(model.tokenizer, prompt).to(model.device)

    logits = choice_logits(model(input_ids).logits, input_ids, model.tokenizer, letters)
    probs = normalized_choice_probs(logits)[0]
    entropy = float(entropy_from_probs(probs.unsqueeze(0), normalized=False).item())
    normalized_entropy = float(entropy / math.log(len(letters))) if len(letters) > 1 else 0.0

    pred_letter = letters[int(torch.argmax(probs).item())]
    ref = str(normalized["answer"]) if normalized["answer"] is not None else None
    ref_prob = float(probs[letters.index(ref)].item()) if ref in letters else None

    obf_letters = _obfuscation_letters(normalized["false_in"], choices)
    obf_prob = None
    if obf_letters:
        obf_prob = float(sum(probs[letters.index(letter)].item() for letter in obf_letters if letter in letters))

    result = {
        "question": normalized["question"],
        "ref": ref,
        "pred": pred_letter,
        "pred_letter": pred_letter,
        "choice_distribution": choice_distribution_dict(probs, letters),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "acc_prob": ref_prob,
        "p_correct": ref_prob,
        "p_obfuscation": obf_prob,
        "obfuscation_letters": obf_letters,
        "choices": choices,
    }
    if normalized["false_in"] is not None:
        result["False_in"] = normalized["false_in"]
    return result


@torch.no_grad()
def evaluate_yes_no_row(
    model: CausalLMWithLoRA,
    row: dict[str, Any],
    *,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Evaluate one Yes/No row using the forced next-token Yes/No distribution.

    Yes/No probes are forced probes.  The score is based on the normalized
    probability of the next token being Yes or No.  Generated refusals are not used
    and no refusal rate is reported for this probe family.
    """

    del max_new_tokens
    normalized = normalize_row(row)
    prompt = yes_no_prompt(str(normalized["question"]))
    input_ids = apply_chat_template(model.tokenizer, prompt).to(model.device)

    next_logits = _logits_at_prompt_end(model, input_ids)
    label_ids = _label_token_ids(model.tokenizer, YES_NO_LABELS).to(model.device)
    label_probs = torch.softmax(next_logits.index_select(dim=-1, index=label_ids), dim=-1)[0]
    entropy = float(entropy_from_probs(label_probs.unsqueeze(0), normalized=False).item())
    normalized_entropy = float(entropy / math.log(2))

    pred_from_probs = "Yes" if label_probs[0].item() >= label_probs[1].item() else "No"
    ref = str(normalized["answer"]).strip().title() if normalized["answer"] is not None else None

    return {
        "question": normalized["question"],
        "ref": ref,
        "pred": pred_from_probs,
        "pred_label": pred_from_probs,
        "yes_no_distribution": {"Yes": float(label_probs[0].item()), "No": float(label_probs[1].item())},
        "p_yes": float(label_probs[0].item()),
        "p_no": float(label_probs[1].item()),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def evaluate_split(
    model: CausalLMWithLoRA,
    people_rows: dict[str, list[dict[str, Any]]],
    *,
    probe_type: str,
    max_new_tokens: int,
    desc: str,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate one named split and return project-format predictions."""

    results: dict[str, list[dict[str, Any]]] = {}
    for name, rows in tqdm(people_rows.items(), desc=desc):
        person_results: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=name, leave=False):
            if probe_type == "mcq":
                person_results.append(evaluate_mcq_row(model, row, max_new_tokens=max_new_tokens))
            elif probe_type == "yes_no":
                person_results.append(evaluate_yes_no_row(model, row, max_new_tokens=max_new_tokens))
            elif probe_type == "open":
                person_results.append(evaluate_open_row(model, row, max_new_tokens=max_new_tokens))
            else:
                raise ValueError(f"Unknown probe_type: {probe_type}")
        results[name] = person_results
    return results


def build_yes_no_splits(
    *,
    yesno_all_file: str | Path | None,
    yesno_more_file: str | Path | None = None,
    obfuscate_samples_file: str | Path | None = None,
    selection: EvaluationSelection | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Build reference, in-training, and out-of-training Yes/No probes.

    The original project used `get_test_set.py` to create these files for WHP runs.  This
    function keeps the same behavior, while also supporting a fallback split that uses
    all available `alternative_in_questions` when no run-specific obfuscation samples
    are supplied.
    """

    splits: dict[str, dict[str, list[dict[str, Any]]]] = {
        "yes_no_reference": {},
        "yes_no_in_training": {},
        "yes_no_out_of_training": {},
    }
    if yesno_all_file is None:
        return splits

    outside_questions = load_json(yesno_all_file)
    more_questions = load_json(yesno_more_file) if yesno_more_file else None
    obfuscate_samples = load_json(obfuscate_samples_file) if obfuscate_samples_file else None

    for key, rows in outside_questions.items():
        if selection is not None and not selection.contains(str(key)):
            continue
        name = selection.resolve_name(str(key)) if selection is not None else str(key)
        splits["yes_no_reference"].setdefault(name, [])
        splits["yes_no_in_training"].setdefault(name, [])
        splits["yes_no_out_of_training"].setdefault(name, [])

        if isinstance(rows, list):
            for row in rows:
                answer_question = row.get("Answer_questions")
                if answer_question:
                    splits["yes_no_reference"][name].append(
                        {"Question": _force_yes_no_instruction(answer_question), "Answer": "Yes", "name": name}
                    )
                for question in row.get("alternative_out_questions", []) or []:
                    splits["yes_no_out_of_training"][name].append(
                        {"Question": _force_yes_no_instruction(question), "Answer": "No", "name": name}
                    )
                # Fallback when run-specific passages are not available.
                if obfuscate_samples is None:
                    for question in row.get("alternative_in_questions", []) or []:
                        splits["yes_no_in_training"][name].append(
                            {"Question": _force_yes_no_instruction(question), "Answer": "No", "name": name}
                        )

        if obfuscate_samples is not None and more_questions is not None:
            passage_list = obfuscate_samples.get(str(key), obfuscate_samples.get(name, []))
            person_more = more_questions.get(str(key), more_questions.get(name, {}))
            if isinstance(person_more, dict):
                for passage in passage_list:
                    for row in person_more.get(passage, []) or []:
                        new_row = dict(row)
                        new_row["Question"] = _force_yes_no_instruction(new_row.get("Question", ""))
                        if "Answer" not in new_row:
                            new_row["Answer"] = "No"
                        new_row.setdefault("name", name)
                        splits["yes_no_in_training"][name].append(new_row)

    return splits


def _force_yes_no_instruction(question: str) -> str:
    q = str(question).strip()
    lowered = q.lower()
    if "yes or no" in lowered:
        return q
    return q.rstrip(" .") + ". Only answer Yes or No."


def score_open_results(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = _all_rows(results)
    return {
        "probe_type": "open",
        "n": len(rows),
        "refusal_rate": _safe_mean([1.0 if row.get("is_refused") else 0.0 for row in rows]),
        "rougeL_recall": _safe_mean([row.get("rougeL_recall") for row in rows]),
    }


def score_mcq_results(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = _all_rows(results)
    accuracy_distribution = []
    accuracy_generation = []
    for row in rows:
        ref = row.get("ref")
        accuracy_distribution.append(1.0 if ref is not None and row.get("pred_letter") == ref else 0.0)
        generated = row.get("generated_letter")
        if generated is not None:
            accuracy_generation.append(1.0 if ref is not None and generated == ref else 0.0)
    return {
        "probe_type": "mcq",
        "n": len(rows),
        "accuracy": _safe_mean(accuracy_distribution),
        "accuracy_distribution": _safe_mean(accuracy_distribution),
        "accuracy_generation": _safe_mean(accuracy_generation),
        "entropy": _safe_mean([row.get("entropy") for row in rows]),
        "normalized_entropy": _safe_mean([row.get("normalized_entropy") for row in rows]),
        "p_correct": _safe_mean([row.get("p_correct", row.get("acc_prob")) for row in rows]),
        "p_obfuscation": _safe_mean([row.get("p_obfuscation") for row in rows]),
    }


def score_yes_no_results(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = _all_rows(results)
    acc_prob = []
    acc_generation = []
    yes_rate = []
    for row in rows:
        ref = row.get("ref")
        pred_label = row.get("pred_label")
        gen_label = row.get("generated_label")
        acc_prob.append(1.0 if ref is not None and pred_label == ref else 0.0)
        if gen_label is not None:
            acc_generation.append(1.0 if ref is not None and gen_label == ref else 0.0)
        yes_rate.append(1.0 if pred_label == "Yes" else 0.0)
    return {
        "probe_type": "yes_no",
        "n": len(rows),
        "accuracy": _safe_mean(acc_prob),
        "accuracy_distribution": _safe_mean(acc_prob),
        "accuracy_generation": _safe_mean(acc_generation),
        "yes_rate": _safe_mean(yes_rate),
        "p_yes": _safe_mean([row.get("p_yes") for row in rows]),
        "entropy": _safe_mean([row.get("entropy") for row in rows]),
        "normalized_entropy": _safe_mean([row.get("normalized_entropy") for row in rows]),
    }


def score_named_split(probe_type: str, results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if probe_type == "open":
        return score_open_results(results)
    if probe_type == "mcq":
        return score_mcq_results(results)
    if probe_type == "yes_no":
        return score_yes_no_results(results)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def _all_rows(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for rows in results.values() for row in rows]


def score_suite(predictions: dict[str, Any]) -> dict[str, Any]:
    """Score all supported splits in a unified prediction dictionary."""

    summary: dict[str, Any] = {}
    for split_name, split_payload in predictions.items():
        if split_name == "metadata":
            continue
        if not isinstance(split_payload, dict):
            continue
        if split_name == "open_ended":
            summary[split_name] = score_named_split("open", split_payload)
        elif split_name == "mcq":
            summary[split_name] = score_named_split("mcq", split_payload)
        elif split_name.startswith("yes_no"):
            summary[split_name] = score_named_split("yes_no", split_payload)
    return summary


def write_summary_csv(summary: dict[str, Any], path: str | Path) -> None:
    """Write summary metrics as a compact CSV table."""

    path = Path(path)
    ensure_dir(path.parent)
    metric_keys = sorted({key for metrics in summary.values() for key in metrics.keys() if key != "probe_type"})
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["split", "probe_type"] + metric_keys)
        writer.writeheader()
        for split_name, metrics in summary.items():
            row = {"split": split_name, **metrics}
            writer.writerow(row)


def evaluate_and_score_suite(
    *,
    model: CausalLMWithLoRA,
    output_dir: str | Path,
    selection: EvaluationSelection | None,
    open_file: str | Path | None = None,
    mcq_file: str | Path | None = None,
    yesno_gt_file: str | Path | None = None,
    yesno_in_file: str | Path | None = None,
    yesno_out_file: str | Path | None = None,
    yesno_all_file: str | Path | None = None,
    yesno_more_file: str | Path | None = None,
    obfuscate_samples_file: str | Path | None = None,
    max_new_tokens_open: int = 64,
    max_new_tokens_mcq: int = 32,
    max_new_tokens_yesno: int = 8,
    run_open: bool = True,
    run_mcq: bool = True,
    run_yesno: bool = True,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all requested forget-set evaluations and write predictions plus scores."""

    output_dir = ensure_dir(output_dir)
    predictions: dict[str, Any] = {"metadata": metadata or {}}

    if run_open and open_file:
        open_rows = flatten_people(load_json(open_file), selection)
        predictions["open_ended"] = evaluate_split(
            model,
            open_rows,
            probe_type="open",
            max_new_tokens=max_new_tokens_open,
            desc="open-ended",
        )

    if run_mcq and mcq_file:
        mcq_rows = flatten_people(load_json(mcq_file), selection)
        predictions["mcq"] = evaluate_split(
            model,
            mcq_rows,
            probe_type="mcq",
            max_new_tokens=max_new_tokens_mcq,
            desc="mcq",
        )

    if run_yesno:
        yesno_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if yesno_gt_file:
            yesno_splits["yes_no_reference"] = flatten_people(load_json(yesno_gt_file), selection)
        if yesno_in_file:
            yesno_splits["yes_no_in_training"] = flatten_people(load_json(yesno_in_file), selection)
        if yesno_out_file:
            yesno_splits["yes_no_out_of_training"] = flatten_people(load_json(yesno_out_file), selection)
        if not yesno_splits and yesno_all_file:
            yesno_splits = build_yes_no_splits(
                yesno_all_file=yesno_all_file,
                yesno_more_file=yesno_more_file,
                obfuscate_samples_file=obfuscate_samples_file,
                selection=selection,
            )
        for split_name, split_rows in yesno_splits.items():
            if _all_rows(split_rows):
                predictions[split_name] = evaluate_split(
                    model,
                    split_rows,
                    probe_type="yes_no",
                    max_new_tokens=max_new_tokens_yesno,
                    desc=split_name,
                )

    summary = score_suite(predictions)
    save_json(predictions, output_dir / "predictions.json")
    save_json(summary, output_dir / "summary.json")
    write_summary_csv(summary, output_dir / "summary.csv")
    return predictions, summary
