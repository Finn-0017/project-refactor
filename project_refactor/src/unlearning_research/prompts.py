"""Prompt builders used by training and evaluation.

Keeping prompts in one module prevents accidental changes between training and
evaluation scripts.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
):
    """Tokenize a single-turn chat prompt."""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )


def mcq_prompt(question: str, choices: dict[str, str], *, answer_letters: list[str] | None = None) -> str:
    """Build the MCQ prompt used for DF-MCQ and probing.

    The answer is constrained to a letter because DF-MCQ optimizes the next-token
    distribution over the choice letters.
    """

    answer_letters = answer_letters or sorted(choices.keys())
    rendered_choices = "\n".join(f"{letter}. {choices[letter]}" for letter in answer_letters)
    letter_list = ", ".join(answer_letters[:-1]) + f" or {answer_letters[-1]}"
    return (
        f"Question:\n{question}\n"
        f"Choose one answer from:\n{rendered_choices}\n"
        f"Respond with ({letter_list}) only."
    )


def whp_generation_prompt(name: str) -> str:
    """Prompt used before the WHP obfuscation passage.

    The labels ignore this prompt and train only on the passage text.
    """

    return f"Generate a passage about {name}."


def open_question_prompt(question: str) -> str:
    """Prompt for open-ended evaluation."""

    return question


def yes_no_prompt(question: str) -> str:
    """Prompt for Yes/No probing."""

    return question.rstrip() + " Answer Yes or No directly."
