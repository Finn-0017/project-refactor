"""Parsing helpers for generated probe answers.

These functions keep the rules for reading short MCQ answers in one place.  The
parser accepts direct answers such as ``B`` and common variants such as
``The answer is B``.  When the model outputs the option text instead of the
letter, the exact text can also be mapped back to its letter.
"""

from __future__ import annotations

import re
from typing import Any


def _normalize_text(text: Any) -> str:
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    return re.sub(r"\s+", " ", str(text).strip())


def extract_mcq_letter(
    text: Any,
    letters: tuple[str, ...] | list[str],
    choices: dict[str, Any] | None = None,
) -> str | None:
    """Extract an option letter from a generated MCQ answer.

    Matching is intentionally simple and auditable:
    1. direct letter answers at the start of the response,
    2. common phrases such as ``answer is B``,
    3. a standalone letter fallback,
    4. an exact or prefix match against the rendered choice text.
    """

    normalized = _normalize_text(text)
    if not normalized:
        return None

    letters_tuple = tuple(str(letter).upper() for letter in letters)
    allowed = "".join(re.escape(letter) for letter in letters_tuple)

    direct_patterns = (
        rf"^\s*(?:answer|option|choice)?\s*[:：=]?\s*[\(\[\{{]?\s*([{allowed}])\s*[\)\]\}}\.\,:：]?\b",
        rf"^\s*([{allowed}])\s*$",
    )
    for pattern in direct_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    phrase_pattern = (
        rf"\b(?:answer|correct answer|option|choice|choose|select|selected)\b"
        rf"(?:\s+(?:is|would be|should be|as))?\s*[:：=]?\s*[\(\[]?\s*([{allowed}])\b"
    )
    match = re.search(phrase_pattern, normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    standalone_pattern = rf"\b([{allowed}])\b"
    match = re.search(standalone_pattern, normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    if choices:
        generated = normalized.lower().strip(" .,:;!?\"'")
        for letter in letters_tuple:
            choice_text = str(choices.get(letter, "")).strip()
            if not choice_text:
                continue
            candidate = choice_text.lower().strip(" .,:;!?\"'")
            if generated == candidate or generated.startswith(candidate) or candidate.startswith(generated):
                return letter

    return None
