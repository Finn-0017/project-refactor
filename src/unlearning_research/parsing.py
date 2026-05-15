"""Parsing helpers for generated probe answers.

MCQ generation can return short forms such as ``B`` and longer forms such as
``The answer is B``.  The parser below keeps this rule in one place so evaluation
outputs are reproducible.
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
    """Extract an MCQ option letter from a short generated answer.

    The matching order starts with explicit letter answers, then natural-language
    answer phrases, then a standalone letter fallback.  If no letter is present, the
    parser can also map an exact generated option text back to its letter.
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
