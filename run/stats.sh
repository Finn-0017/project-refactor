#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run/stats.sh whp 32 20
#   bash run/stats.sh dfmcq 32 20
#   bash run/stats.sh orig

METHOD="${1:-}"
LORA_RANK="${2:-}"
NUM_SAMPLES="${3:-}"

if [ -z "$METHOD" ]; then
  echo "Usage: bash run/stats.sh whp 32 20"
  echo "       bash run/stats.sh dfmcq 32 20"
  echo "       bash run/stats.sh orig"
  exit 1
fi

if [ "$METHOD" != "orig" ] && [ "$METHOD" != "orig_model" ]; then
  if [ -z "$LORA_RANK" ] || [ -z "$NUM_SAMPLES" ]; then
    echo "Missing arguments. Example: bash run/stats.sh $METHOD 32 20"
    exit 1
  fi
fi

python - "$METHOD" "$LORA_RANK" "$NUM_SAMPLES" <<'PY'
import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

method = sys.argv[1]
lora_rank = sys.argv[2]
num_samples = sys.argv[3]
exp_root = Path("exp")
sets = {1, 2, 3, 4, 5}

seed_re = re.compile(r"seed(\d+)")

@dataclass(frozen=True)
class EvalDir:
    seed: str
    set_id: int
    path: Path


def to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_numeric_column(rows, column):
    for row in rows:
        value = str(row.get(column, "")).strip()
        if value == "":
            continue
        if to_float(value) is None:
            return False
    return True


def count_columns(columns):
    return [col for col in columns if col == "n" or col.endswith("_n")]


def group_columns(rows):
    if not rows:
        return []
    columns = list(rows[0].keys())
    known = {"split", "probe_type"}
    groups = [col for col in columns if col in known]
    for col in columns:
        if col in known or col == "run_name":
            continue
        values = [str(row.get(col, "")).strip() for row in rows]
        non_empty = [value for value in values if value]
        if non_empty and not is_numeric_column(rows, col):
            groups.append(col)
    return groups


def choose_weight_column(metric, counts):
    if not counts:
        return None
    if "n" in counts:
        return "n"

    prefixes = sorted(((col[:-2], col) for col in counts), key=lambda x: len(x[0]), reverse=True)
    for prefix, col in prefixes:
        if metric == prefix or metric.startswith(prefix + "_"):
            return col

    if metric.startswith("forget_") and "forget_n" in counts:
        return "forget_n"
    if metric.startswith("hardretain_") and "hardretain_n" in counts:
        return "hardretain_n"
    if metric.startswith("retain_") and "retain_n" in counts:
        return "retain_n"
    return counts[0] if len(counts) == 1 else None


def aggregate_rows(rows):
    if not rows:
        return []
    columns = list(rows[0].keys())
    groups = group_columns(rows)
    counts = count_columns(columns)
    numeric = [col for col in columns if col not in groups and is_numeric_column(rows, col)]
    metrics = [col for col in numeric if col not in counts]

    by_key = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, "") for col in groups) if groups else ("all",)
        by_key[key].append(row)

    output = []
    for key, chunk in by_key.items():
        row_out = {}
        if groups:
            for col, value in zip(groups, key):
                row_out[col] = value

        for col in counts:
            total = 0.0
            seen = False
            for row in chunk:
                value = to_float(row.get(col))
                if value is not None:
                    total += value
                    seen = True
            if seen:
                row_out[col] = int(total) if total.is_integer() else total

        for col in metrics:
            weight_col = choose_weight_column(col, counts)
            numerator = 0.0
            denominator = 0.0
            values = []
            for row in chunk:
                value = to_float(row.get(col))
                if value is None:
                    continue
                if weight_col:
                    weight = to_float(row.get(weight_col))
                    if weight is None or weight <= 0:
                        continue
                    numerator += value * weight
                    denominator += weight
                else:
                    values.append(value)
            if denominator > 0:
                row_out[col] = numerator / denominator
            elif values:
                row_out[col] = sum(values) / len(values)
            else:
                row_out[col] = ""
        output.append(row_out)
    return output


def discover_eval_dirs():
    result = []
    if method in {"orig", "orig_model"}:
        for root in [exp_root / "orig_model", exp_root / "orig_model_eval"]:
            if not root.exists():
                continue
            for set_id in sorted(sets):
                for candidate in [root / f"set{set_id}" / "eval", root / f"set{set_id}"]:
                    if (candidate / "summary.csv").exists():
                        result.append(EvalDir("orig", set_id, candidate))
                        break
        return result

    pattern = str(exp_root / method / f"n{num_samples}_lora{lora_rank}_seed*")
    for seed_root in sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir()):
        match = seed_re.search(seed_root.name)
        seed = match.group(1) if match else seed_root.name
        for set_id in sorted(sets):
            for candidate in [seed_root / f"set{set_id}" / "eval", seed_root / f"set{set_id}"]:
                if (candidate / "summary.csv").exists():
                    result.append(EvalDir(seed, set_id, candidate))
                    break
    return result


def display_key(row):
    keys = [col for col in ["split", "probe_type"] if col in row]
    if not keys:
        return "all"
    return "|".join(f"{col}={row.get(col, '')}" for col in keys)


def flatten(seed, file_name, rows):
    flat = []
    for row in rows:
        key = display_key(row)
        for col, value in row.items():
            number = to_float(value)
            if number is None:
                continue
            if col == "n" or col.endswith("_n"):
                continue
            flat.append({
                "seed": seed,
                "file": file_name,
                "row": key,
                "metric": col,
                "value": number,
            })
    return flat


def sample_variance(values):
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def build_report(flat_rows):
    grouped = defaultdict(list)
    for row in flat_rows:
        grouped[(row["file"], row["row"], row["metric"])].append(float(row["value"]))
    report = []
    for (file_name, row_key, metric), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        var = sample_variance(values)
        report.append({
            "file": file_name,
            "row": row_key,
            "metric": metric,
            "n_seeds": len(values),
            "mean": mean,
            "variance": var,
            "std": math.sqrt(var),
            "min": min(values),
            "max": max(values),
        })
    return report


def key_report_rows(report):
    wanted = {
        ("brian_table_open.csv", "all", "forget_refusal_rate"),
        ("brian_table_open.csv", "all", "forget_rougeL_recall"),
        ("brian_table_open.csv", "all", "retain_rougeL_recall"),
        ("brian_table_open.csv", "all", "hardretain_rougeL_recall"),
        ("brian_table_mcq.csv", "all", "forget_accuracy"),
        ("brian_table_mcq.csv", "all", "forget_entropy"),
        ("brian_table_mcq.csv", "all", "forget_p_obfuscation"),
        ("brian_table_mcq.csv", "all", "hardretain_accuracy"),
        ("brian_table_yesno.csv", "all", "forget_ref_yes_rate"),
        ("brian_table_yesno.csv", "all", "forget_false_yes_rate"),
        ("brian_table_yesno.csv", "all", "retain_accuracy"),
        ("brian_table_yesno.csv", "all", "hardretain_accuracy"),
    }
    return [row for row in report if (row["file"], row["row"], row["metric"]) in wanted]


if method in {"orig", "orig_model"}:
    output_dir = Path("exp/stats/orig_model")
else:
    output_dir = Path(f"exp/stats/{method}_n{num_samples}_lora{lora_rank}")

output_dir.mkdir(parents=True, exist_ok=True)
eval_dirs = discover_eval_dirs()
if not eval_dirs:
    if method in {"orig", "orig_model"}:
        print("No CSVs found. Checked exp/orig_model and exp/orig_model_eval.")
    else:
        print(f"No CSVs found. Checked exp/{method}/n{num_samples}_lora{lora_rank}_seed*/set*/eval.")
    raise SystemExit(1)

manifest = [{"seed": e.seed, "set": e.set_id, "eval_dir": str(e.path)} for e in sorted(eval_dirs, key=lambda x: (x.seed, x.set_id))]
write_csv(output_dir / "input_manifest.csv", manifest)

all_flat = []
seeds = sorted({e.seed for e in eval_dirs}, key=lambda x: (x != "orig", x))
for seed in seeds:
    seed_dirs = [e for e in eval_dirs if e.seed == seed]
    found = {e.set_id for e in seed_dirs}
    missing = sorted(sets - found)
    if missing:
        print(f"Warning: seed {seed} missing sets {missing}")

    files = defaultdict(list)
    for e in seed_dirs:
        for path in e.path.glob("*.csv"):
            files[path.name].append(path)

    for file_name, paths in sorted(files.items()):
        rows = []
        for path in paths:
            rows.extend(read_csv(path))
        aggregated = aggregate_rows(rows)
        write_csv(output_dir / f"seed_{seed}" / file_name, aggregated)
        all_flat.extend(flatten(seed, file_name, aggregated))

write_csv(output_dir / "seed_summary_long.csv", all_flat)
report = build_report(all_flat)
write_csv(output_dir / "report_mean_variance.csv", report)
key_rows = key_report_rows(report)
write_csv(output_dir / "report_key_metrics.csv", key_rows)

print(f"Found {len(eval_dirs)} eval directories across {len(seeds)} seed(s).")
print(f"Output: {output_dir}")
print(f"All-set per seed: {output_dir}/seed_<seed>/*.csv")
print(f"Full report: {output_dir}/report_mean_variance.csv")
print(f"Key report: {output_dir}/report_key_metrics.csv")

if key_rows:
    print("\nKey metrics")
    print("metric,mean,variance,std,n_seeds")
    for row in key_rows:
        print(f"{row['file']}:{row['metric']},{float(row['mean']):.6f},{float(row['variance']):.6f},{float(row['std']):.6f},{row['n_seeds']}")
PY
