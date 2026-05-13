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


def get_choice_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
) -> torch.Tensor:
    """Return token IDs corresponding to answer letters.

    The original scripts used tokenizer-specific indexing rules. This helper instead asks
    the tokenizer for the tokenization without special tokens and takes the final token.
    This works for tokenizers that prepend BOS only when special tokens are requested.
    """

    token_ids: list[int] = []
    for letter in letters:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Could not tokenize choice letter {letter!r}")
        token_ids.append(ids[-1])
    return torch.tensor(token_ids, dtype=torch.long)


def last_nonpad_indices(input_ids: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    """Return the final non-padding position for each sequence in a batch."""

    if pad_token_id is None:
        return torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
    lengths = input_ids.ne(pad_token_id).sum(dim=1)
    return torch.clamp(lengths - 1, min=0)


def logits_at_prompt_end(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Select logits for the next token after each prompt.

    Causal LM logits at position `t` predict the next token after token `t`. Since MCQ
    prompts end immediately before the answer letter, the final non-padding position is
    the position used for the choice distribution.
    """

    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    cols = last_nonpad_indices(input_ids, pad_token_id)
    return logits[rows, cols]


def choice_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
) -> torch.Tensor:
    """Return logits restricted to the available choice letters."""

    full_next_token_logits = logits_at_prompt_end(logits, input_ids, tokenizer.pad_token_id)
    indices = get_choice_token_ids(tokenizer, tuple(letters)).to(full_next_token_logits.device)
    return full_next_token_logits.index_select(dim=-1, index=indices)


def normalized_choice_probs(choice_logits_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize probabilities over the choice set only."""

    return torch.softmax(choice_logits_tensor, dim=-1)


def entropy_from_probs(probs: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
    """Compute entropy for a probability distribution.

    If `normalized=True`, entropy is divided by log(K), where K is the number of choices.
    """

    eps = torch.finfo(probs.dtype).eps
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    if normalized:
        entropy = entropy / math.log(probs.size(-1))
    return entropy


def choice_distribution_dict(
    probs: torch.Tensor,
    letters: tuple[str, ...] | list[str] = DEFAULT_CHOICE_LETTERS,
) -> dict[str, float]:
    """Convert a single probability vector to a JSON-friendly dictionary."""

    values = probs.detach().cpu().float().tolist()
    return {letter: float(values[i]) for i, letter in enumerate(letters)}
