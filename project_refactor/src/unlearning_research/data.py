"""Dataset classes and deterministic data-selection helpers."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

from .prompts import apply_chat_template, mcq_prompt, whp_generation_prompt
from .utils import load_json, load_selected_ids


@dataclass(frozen=True)
class MCQExample:
    """A multiple-choice question used for DF-MCQ training or MCQ probing."""

    person_id: str
    name: str
    question: str
    choices: dict[str, str]
    answer: str | None = None


@dataclass(frozen=True)
class WHPExample:
    """A supervised WHP-style obfuscation example."""

    person_id: str
    name: str
    passage_id: int
    passage: str


def load_name_table(names_path: str | Path) -> dict[str, str]:
    """Load `{id: name}` mapping from `whp_names.json`."""

    rows = load_json(names_path)
    mapping: dict[str, str] = {}
    for row in rows:
        if "id" in row and "name" in row:
            mapping[str(row["id"])] = row["name"]
    return mapping


def load_mcq_examples(
    data_path: str | Path,
    selected_ids: list[str],
) -> tuple[list[MCQExample], list[MCQExample]]:
    """Load forget and retain MCQ examples from the current project JSON format."""

    data = load_json(data_path)
    forget: list[MCQExample] = []
    retain: list[MCQExample] = []

    for person_id, rows in data.items():
        if not rows:
            continue
        destination = forget if str(person_id) in selected_ids else retain
        for row in rows:
            choices = row.get("choices") or row.get("Choices")
            if not isinstance(choices, dict):
                raise ValueError(f"Missing choices for MCQ row under person id {person_id}")
            destination.append(
                MCQExample(
                    person_id=str(person_id),
                    name=str(row.get("name", "")),
                    question=str(row.get("question", row.get("Question", ""))),
                    choices={str(k): str(v) for k, v in choices.items()},
                    answer=row.get("answer", row.get("Answer")),
                )
            )

    if not forget:
        raise ValueError(f"No forget MCQs found for selected IDs {selected_ids}")
    if not retain:
        raise ValueError("No retain MCQs found. Check selected IDs and training data.")
    return forget, retain


def stable_person_seed(seed: int, name: str) -> int:
    """Create a deterministic per-person seed without relying on Python's hash salt."""

    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def nested_indices(pool_size: int, num_items: int, *, seed: int, name: str) -> list[int]:
    """Return deterministic nested indices for a person's passage pool.

    For a fixed `(seed, name, pool_size)`, the first 20 indices are always a subset of
    the first 50 indices. This property is required for clean data-amount sweeps.
    """

    if num_items > pool_size:
        raise ValueError(f"Requested {num_items} passages, but the pool only has {pool_size}")
    rng = random.Random(stable_person_seed(seed, name))
    order = list(range(pool_size))
    rng.shuffle(order)
    return order[:num_items]


def load_whp_examples(
    *,
    names_path: str | Path,
    selected_ids: str | Path | list[str] | int,
    obfuscate_passages_path: str | Path,
    num_passages: int,
    seed: int,
    explicit_passage_ids: list[int] | None = None,
) -> tuple[list[WHPExample], dict[str, Any]]:
    """Load WHP examples from precomputed obfuscation passages."""

    selected = load_selected_ids(selected_ids)
    id_to_name = load_name_table(names_path)
    passages_by_name = load_json(obfuscate_passages_path)

    examples: list[WHPExample] = []
    selected_manifest: dict[str, Any] = {}
    for person_id in selected:
        if person_id not in id_to_name:
            raise ValueError(f"Selected id {person_id} is not present in {names_path}")
        name = id_to_name[person_id]
        if name not in passages_by_name:
            raise ValueError(f"No obfuscation passages found for {name!r}")
        pool = passages_by_name[name]
        if explicit_passage_ids is None:
            passage_ids = nested_indices(len(pool), num_passages, seed=seed, name=name)
        else:
            passage_ids = explicit_passage_ids
            if max(passage_ids, default=-1) >= len(pool):
                raise ValueError(f"Passage id out of range for {name!r}: pool size {len(pool)}")
        selected_manifest[name] = {
            "person_id": person_id,
            "pool_size": len(pool),
            "num_passages": len(passage_ids),
            "passage_ids": passage_ids,
        }
        for passage_id in passage_ids:
            examples.append(
                WHPExample(
                    person_id=person_id,
                    name=name,
                    passage_id=passage_id,
                    passage=str(pool[passage_id]),
                )
            )
    return examples, selected_manifest


class DFMCQDataset(Dataset):
    """Dataset for DF-MCQ training.

    Each item contains one forget MCQ and one retain MCQ. The retain MCQ is sampled from
    a seeded RNG so that the sequence is reproducible while still varying across steps.
    """

    def __init__(
        self,
        forget_examples: list[MCQExample],
        retain_examples: list[MCQExample],
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int = 1,
    ) -> None:
        self.forget_examples = forget_examples
        self.retain_examples = retain_examples
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.forget_examples)

    def _encode_mcq(self, example: MCQExample) -> torch.Tensor:
        letters = sorted(example.choices.keys())
        prompt = mcq_prompt(example.question, example.choices, answer_letters=letters)
        return apply_chat_template(self.tokenizer, prompt)[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        forget = self.forget_examples[index]
        retain = self.rng.choice(self.retain_examples)
        return {
            "forget_input_ids": self._encode_mcq(forget),
            "retain_input_ids": self._encode_mcq(retain),
            "forget_name": forget.name,
            "retain_name": retain.name,
        }


class WHPPrecomputedDataset(Dataset):
    """Dataset for WHP training using precomputed obfuscation passages."""

    def __init__(self, examples: list[WHPExample], tokenizer: PreTrainedTokenizerBase) -> None:
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        prompt_ids = apply_chat_template(
            self.tokenizer,
            whp_generation_prompt(example.name),
        )[0]
        passage_ids = self.tokenizer(
            example.passage,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        input_ids = torch.cat([prompt_ids, passage_ids], dim=0)
        labels = torch.cat([
            torch.full_like(prompt_ids, -100),
            passage_ids,
        ])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "name": example.name,
            "person_id": example.person_id,
            "passage_id": example.passage_id,
        }


def collate_dfmcq(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    return {
        "forget_input_ids": pad_sequence(
            [item["forget_input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "retain_input_ids": pad_sequence(
            [item["retain_input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "forget_name": [item["forget_name"] for item in batch],
        "retain_name": [item["retain_name"] for item in batch],
    }


def collate_whp(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    return {
        "input_ids": pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "labels": pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        ),
        "name": [item["name"] for item in batch],
        "person_id": [item["person_id"] for item in batch],
        "passage_id": [item["passage_id"] for item in batch],
    }
