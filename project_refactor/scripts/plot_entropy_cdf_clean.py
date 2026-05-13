#!/usr/bin/env python
"""Plot CDFs of normalized MCQ entropy for refused vs. non-refused items."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.utils import load_json


def iter_items(data):
    for person_items in data.values():
        if isinstance(person_items, dict):
            for nested_items in person_items.values():
                if isinstance(nested_items, list):
                    yield from nested_items
        elif isinstance(person_items, list):
            yield from person_items


def collect(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    refused = []
    answered = []
    for path in paths:
        data = load_json(path)
        for item in iter_items(data):
            entropy = item.get("normalized_entropy", item.get("entropy"))
            if entropy is None:
                continue
            if item.get("is_refused", False):
                refused.append(float(entropy))
            else:
                answered.append(float(entropy))
    return np.asarray(refused), np.asarray(answered)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.asarray([]), np.asarray([])
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot refused/non-refused entropy CDF")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="entropy_cdf_clean.png")
    args = parser.parse_args()

    refused, answered = collect(args.inputs)
    plt.figure(figsize=(5, 4))
    for values, label in [(refused, "refused"), (answered, "not refused")]:
        x, y = ecdf(values)
        if len(x):
            plt.step(x, y, where="post", label=label, linewidth=2)
    plt.xlabel("Normalized MCQ entropy")
    plt.ylabel("CDF")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"refused={len(refused)} not_refused={len(answered)} output={args.output}")


if __name__ == "__main__":
    main()
