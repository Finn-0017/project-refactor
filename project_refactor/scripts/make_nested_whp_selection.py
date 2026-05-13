#!/usr/bin/env python
"""Create deterministic nested WHP passage selections for data-amount sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlearning_research.data import load_name_table, nested_indices
from unlearning_research.utils import ensure_dir, load_json, load_selected_ids, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create nested WHP passage IDs")
    parser.add_argument("--names_path", required=True)
    parser.add_argument("--obfuscate_passages", required=True)
    parser.add_argument("--selected_ids", required=True)
    parser.add_argument("--sizes", required=True, help="Comma-separated sizes, e.g. 5,10,20,50,100,200")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    selected_ids = load_selected_ids(args.selected_ids)
    id_to_name = load_name_table(args.names_path)
    passages = load_json(args.obfuscate_passages)
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    full_manifest = {}
    for person_id in selected_ids:
        name = id_to_name[person_id]
        pool = passages[name]
        person_manifest = {}
        for size in sizes:
            person_manifest[str(size)] = nested_indices(len(pool), size, seed=args.seed, name=name)
        full_manifest[name] = {
            "person_id": person_id,
            "pool_size": len(pool),
            "seed": args.seed,
            "sizes": person_manifest,
        }

    save_json(full_manifest, output_dir / f"nested_whp_selection_seed{args.seed}.json")


if __name__ == "__main__":
    main()
