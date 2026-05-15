#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run/eval_orig_model.sh 0
#   bash run/eval_orig_model.sh 0 all
#   bash run/eval_orig_model.sh 0 3
#   bash run/eval_orig_model.sh 0 all 20
#
# Args:
#   1 = physical GPU id
#   2 = forget set id, or "all" for set1-set5. Default: all
#   3 = number of retain people to keep in the fast retain split. Default: 20

GPU_ID="${1:-0}"
SET_ID_ARG="${2:-all}"
FAST_RETAIN_PEOPLE="${3:-20}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"

BASE_DATA_DIR="data/whp_probe"
NAMES_PATH="data/whp_names/names_210.json"
LORA_CONFIG="${LORA_CONFIG:-configs/lora_rank_256.json}"

SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW_TOKENS_OPEN="${MAX_NEW_TOKENS_OPEN:-64}"
MAX_NEW_TOKENS_MCQ="${MAX_NEW_TOKENS_MCQ:-8}"
MAX_NEW_TOKENS_YESNO="${MAX_NEW_TOKENS_YESNO:-8}"

MAX_RETRY="${MAX_RETRY:-5}"
RETRY_WAIT_SECONDS="${RETRY_WAIT_SECONDS:-1}"

FAST_DATA_DIR="${BASE_DATA_DIR}_fast${FAST_RETAIN_PEOPLE}_seed${SEED}"

if [ ! -d "$BASE_DATA_DIR" ]; then
  echo "Missing data directory: $BASE_DATA_DIR"
  exit 1
fi

if [ ! -f "$NAMES_PATH" ]; then
  echo "Missing names file: $NAMES_PATH"
  exit 1
fi

if [ ! -f "$LORA_CONFIG" ]; then
  echo "Missing LoRA config: $LORA_CONFIG"
  exit 1
fi

# Build a fast probe directory. All files are symlinked from BASE_DATA_DIR,
# except retain-style files, which are sampled by person to reduce eval time.
python - "$BASE_DATA_DIR" "$FAST_DATA_DIR" "$FAST_RETAIN_PEOPLE" "$SEED" <<'PY'
import json
import os
import random
import shutil
import sys
from pathlib import Path

base = Path(sys.argv[1]).resolve()
fast = Path(sys.argv[2]).resolve()
n_people = int(sys.argv[3])
seed = int(sys.argv[4])

fast.mkdir(parents=True, exist_ok=True)

# These files are sampled if they exist. Other probe files are left unchanged.
sampled_names = {
    "whp_unlearn_testset_retain.json",
    "whp_unlearn_testset_retain_yesno.json",
    "whp_unlearn_testset_retain_mcq.json",
}

manifest = {
    "base_data_dir": str(base),
    "fast_data_dir": str(fast),
    "seed": seed,
    "n_people": n_people,
    "files": {},
}

for src in sorted(base.iterdir()):
    if not src.is_file():
        continue

    dst = fast / src.name

    if src.name in sampled_names:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            keys = sorted(data.keys())
            rng = random.Random(seed)
            selected = sorted(rng.sample(keys, min(n_people, len(keys))))
            sampled = {key: data[key] for key in selected}

            with dst.open("w", encoding="utf-8") as f:
                json.dump(sampled, f, indent=2, ensure_ascii=False)

            manifest["files"][src.name] = {
                "mode": "sampled_by_person",
                "original_people": len(keys),
                "selected_people": len(selected),
                "selected_names": selected,
            }
            continue

        # If the structure is not person -> questions, keep the full file.
        shutil.copy2(src, dst)
        manifest["files"][src.name] = {"mode": "copied_unrecognized_structure"}
        continue

    # Keep non-retain files as symlinks. Replace stale files if necessary.
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)
    manifest["files"][src.name] = {"mode": "symlink"}

with (fast / "fast_retain_manifest.json").open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
PY

REQUIRED_DATA_FILES=(
  "whp_unlearn_testset_forget.json"
  "whp_unlearn_testset_forget_mcq.json"
  "whp_unlearn_testset_forget_yesno.json"
  "whp_unlearn_testset_forget_yesno_false_control.json"
  "whp_unlearn_testset_hardretain.json"
  "whp_unlearn_testset_hardretain_mcq.json"
  "whp_unlearn_testset_hardretain_yesno.json"
  "whp_unlearn_testset_retain.json"
  "whp_unlearn_testset_retain_yesno.json"
)

for file_name in "${REQUIRED_DATA_FILES[@]}"; do
  if [ ! -f "$FAST_DATA_DIR/$file_name" ]; then
    echo "Missing probe file: $FAST_DATA_DIR/$file_name"
    exit 1
  fi
done

if [ "$SET_ID_ARG" = "all" ]; then
  SET_IDS=(1 2 3 4 5)
else
  SET_IDS=("$SET_ID_ARG")
fi

echo "Using physical GPU: $GPU_ID"
echo "Model path: $MODEL_PATH"
echo "Base data directory: $BASE_DATA_DIR"
echo "Fast data directory: $FAST_DATA_DIR"
echo "Fast retain people: $FAST_RETAIN_PEOPLE"
echo "Names file: $NAMES_PATH"
echo "LoRA config: $LORA_CONFIG"
echo "Sets: ${SET_IDS[*]}"
echo

eval_one_set() {
  local set_id="$1"
  local selected_ids="configs/unlearn_ids${set_id}.json"
  local output_dir="exp/orig_model_fast${FAST_RETAIN_PEOPLE}/set${set_id}/eval"
  local run_name="initial_model_set${set_id}_fast${FAST_RETAIN_PEOPLE}"

  if [ ! -f "$selected_ids" ]; then
    echo "Missing selected ids file: $selected_ids"
    exit 1
  fi

  mkdir -p "$output_dir"

  echo "============================================================"
  echo "Evaluating set${set_id}"
  echo "Selected ids file: $selected_ids"
  echo "Output directory: $output_dir"
  echo "Selected ids:"
  cat "$selected_ids"
  echo

  local attempt=1
  while true; do
    echo
    echo "Starting initial model WPU probe evaluation for set${set_id}, attempt ${attempt}/${MAX_RETRY}"

    if python scripts/eval_wpu_probe_clean.py \
      --base_model_path "$MODEL_PATH" \
      --origmodel \
      --lora_config "$LORA_CONFIG" \
      --data_dir "$FAST_DATA_DIR" \
      --selected_ids "$selected_ids" \
      --names_path "$NAMES_PATH" \
      --output_dir "$output_dir" \
      --run_name "$run_name" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --max_new_tokens_open "$MAX_NEW_TOKENS_OPEN" \
      --max_new_tokens_mcq "$MAX_NEW_TOKENS_MCQ" \
      --max_new_tokens_yesno "$MAX_NEW_TOKENS_YESNO"; then

      echo
      echo "Initial model WPU probe evaluation finished for set${set_id}."
      echo "Output directory: $output_dir"
      echo "Main summary: $output_dir/summary.csv"
      echo "Open table: $output_dir/brian_table_open.csv"
      echo "MCQ table: $output_dir/brian_table_mcq.csv"
      echo "YesNo table: $output_dir/brian_table_yesno.csv"
      break
    fi

    if [ "$attempt" -ge "$MAX_RETRY" ]; then
      echo
      echo "Initial model WPU probe evaluation failed for set${set_id} after ${MAX_RETRY} attempts."
      exit 1
    fi

    echo
    echo "Evaluation failed for set${set_id}. Retrying in ${RETRY_WAIT_SECONDS} seconds..."
    sleep "$RETRY_WAIT_SECONDS"
    attempt=$((attempt + 1))
  done
}

for set_id in "${SET_IDS[@]}"; do
  eval_one_set "$set_id"
done

echo
echo "All requested original-model evaluations finished."
echo "Fast data manifest: $FAST_DATA_DIR/fast_retain_manifest.json"
