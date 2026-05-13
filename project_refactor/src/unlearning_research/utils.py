"""General utilities shared by training and evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common random seeds used by data sampling and model training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as fin:
        return json.load(fin)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Save JSON with stable formatting."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(_to_jsonable(data), fout, indent=indent, ensure_ascii=False)


def append_log(message: str, logfile: str | Path | None = None, *, also_print: bool = True) -> None:
    """Write a log message to stdout and optionally to a file."""

    if also_print:
        print(message, flush=True)
    if logfile is not None:
        logfile = Path(logfile)
        ensure_dir(logfile.parent)
        with logfile.open("a", encoding="utf-8") as fout:
            fout.write(message + "\n")


def file_sha256(path: str | Path) -> str:
    """Compute a SHA-256 hash for a file.

    The hash is useful for proving that two runs used the same data source.
    """

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_exists(src: str | Path | None, dst_dir: str | Path) -> str | None:
    """Copy a configuration file into an output directory when it is available."""

    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_dir = ensure_dir(dst_dir)
    dst_path = dst_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def load_selected_ids(selected_ids: str | Path | int | list[str] | list[int]) -> list[str]:
    """Normalize selected target IDs to strings.

    The existing data files commonly use string keys for person IDs, while shell scripts
    sometimes pass integers. Normalizing early avoids silent mismatches.
    """

    if isinstance(selected_ids, (list, tuple)):
        return [str(x) for x in selected_ids]
    if isinstance(selected_ids, int):
        return [str(selected_ids)]

    path = Path(str(selected_ids))
    if path.exists():
        loaded = load_json(path)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a list of selected IDs in {path}, got {type(loaded)}")
        return [str(x) for x in loaded]

    if "," in str(selected_ids):
        return [part.strip() for part in str(selected_ids).split(",") if part.strip()]
    return [str(selected_ids)]


def build_file_manifest(paths: dict[str, str | Path | None]) -> dict[str, Any]:
    """Return file paths and hashes for reproducibility metadata."""

    manifest: dict[str, Any] = {}
    for key, value in paths.items():
        if value is None:
            manifest[key] = None
            continue
        path = Path(value)
        manifest[key] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
        }
    return manifest


def _to_jsonable(value: Any) -> Any:
    """Convert common Python and tensor objects into JSON-serializable values."""

    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_int_list(value: str | None) -> list[int] | None:
    """Parse comma-separated integer lists used for passage IDs."""

    if value is None or value == "" or value == "-1":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move all tensors in a batch dictionary to the target device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def parameter_l2_norm(parameters) -> float:
    """Compute the L2 norm over parameters or gradients."""

    total = 0.0
    for param in parameters:
        if param is None:
            continue
        data = param.detach()
        total += float(data.norm(2).item() ** 2)
    return total ** 0.5


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Return a compact summary of trainable parameters."""

    total = 0
    trainable = 0
    names = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / max(total, 1),
        "trainable_tensors": names,
    }
