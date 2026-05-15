"""Choice-letter probability and entropy utilities."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any


DEFAULT_CHOICE_LETTERS = ("A", "B", "C", "D", "E")


def _use_qwen_rule(tokenizer: PreTrainedTokenizerBase, model_path: str | None = None) -> bool:
    """Return whether legacy Qwen-style choice-token indexing should be used."""

    candidates = [model_path or ""]
    for attr in ("name_or_path",):
        value = getattr(tokenizer, attr, "")
        if value:
            candidates.append(str(value))
    return any("qwen" in item.lower() for item in candidates)


def get_legacy_choice_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
    *,
    model_path: str | None = None,
) -> torch.Tensor:
    """Return choice-token IDs compatible with the original evaluation scripts.

    The previous code used ``tokenizer.encode(letter)`` with default tokenizer settings,
    then selected index 1 for Llama-style tokenizers and index 0 for Qwen-style
    tokenizers.  Using the same rule in training and evaluation avoids hidden token-ID
    mismatches.
    """

    use_qwen_rule = _use_qwen_rule(tokenizer, model_path)
    token_ids: list[int] = []
    for letter in letters:
        ids = tokenizer.encode(letter)
        if not ids:
            raise ValueError(f"Could not tokenize choice letter {letter!r}")
        index = 0 if use_qwen_rule else 1
        if len(ids) <= index:
            index = len(ids) - 1
        token_ids.append(ids[index])
    return torch.tensor(token_ids, dtype=torch.long)


def get_choice_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
    *,
    model_path: str | None = None,
) -> torch.Tensor:
    """Return answer-letter token IDs.

    This intentionally delegates to the legacy rule so DF-MCQ training and MCQ probing
    read out the same choice tokens.
    """

    return get_legacy_choice_token_ids(tokenizer, letters, model_path=model_path)


def last_real_token_indices(
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Return the final real-token position for each sequence in a batch.

    Prefer an explicit attention mask produced by the collator.  Falling back to
    ``input_ids != pad_token_id`` is only safe when pad and EOS are different.
    """

    if attention_mask is not None:
        lengths = attention_mask.long().sum(dim=1)
        return torch.clamp(lengths - 1, min=0)
    # Evaluation calls usually contain one unpadded prompt.  Do not infer padding from
    # token values because pad may equal EOS and EOS/EOT may appear inside chat prompts.
    del pad_token_id
    return torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)


def logits_at_prompt_end(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int | None = None,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_end_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select logits for the next token after each prompt."""

    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    if prompt_end_indices is None:
        cols = last_real_token_indices(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
        )
    else:
        cols = prompt_end_indices.to(input_ids.device)
    return logits[rows, cols]


def choice_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_end_indices: torch.Tensor | None = None,
    model_path: str | None = None,
) -> torch.Tensor:
    """Return logits restricted to the available choice letters."""

    full_next_token_logits = logits_at_prompt_end(
        logits,
        input_ids,
        tokenizer.pad_token_id,
        attention_mask=attention_mask,
        prompt_end_indices=prompt_end_indices,
    )
    indices = get_choice_token_ids(tokenizer, tuple(letters), model_path=model_path).to(full_next_token_logits.device)
    return full_next_token_logits.index_select(dim=-1, index=indices)


def normalized_choice_probs(choice_logits_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize probabilities over the choice set only."""

    return torch.softmax(choice_logits_tensor, dim=-1)


def raw_full_vocab_choice_probs(
    next_token_logits: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
    *,
    model_path: str | None = None,
) -> torch.Tensor:
    """Return full-vocabulary softmax probabilities sliced to choice letters.

    This matches the original project logic: normalize over the whole vocabulary, then
    read out A/B/C/D/E.  The returned values generally do not sum to one.
    """

    indices = get_legacy_choice_token_ids(tokenizer, tuple(letters), model_path=model_path).to(next_token_logits.device)
    full_probs = torch.softmax(next_token_logits, dim=-1)
    return full_probs.index_select(dim=-1, index=indices)


def entropy_from_probs(probs: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
    """Compute entropy for a probability distribution."""

    eps = torch.finfo(probs.dtype).eps
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    if normalized:
        entropy = entropy / math.log(probs.size(-1))
    return entropy


def entropy_from_unnormalized_probs(probs: torch.Tensor) -> torch.Tensor:
    """Compute the legacy entropy on unnormalized sliced probabilities."""

    return entropy_from_probs(probs, normalized=False)


def choice_distribution_dict(
    probs: torch.Tensor,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
) -> dict[str, float]:
    """Convert a single probability vector to a JSON-friendly dictionary."""

    values = probs.detach().cpu().float().tolist()
    return {letter: float(values[i]) for i, letter in enumerate(letters)}
