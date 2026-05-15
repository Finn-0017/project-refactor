#!/usr/bin/env bash
set -euo pipefail

# Calibration for the current initial/base model without loading a trained LoRA checkpoint.
# Example:
#   SET_ID=1 BASE_MODEL_PATH=/path/to/modified_llama LORA_CONFIG=config/lora_config10.json bash scripts/eval_wpu_probe_initial_one_set.sh

SET_ID="${SET_ID:-1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-data/whp_probe}"
NAMES_PATH="${NAMES_PATH:-data/WHPplus/whp_names.json}"
SELECTED_IDS="${SELECTED_IDS:-config/unlearn_ids${SET_ID}.json}"
LORA_CONFIG="${LORA_CONFIG:-config/lora_config10.json}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/initial_model_eval_set${SET_ID}}"
RUN_NAME="${RUN_NAME:-initial_model_set${SET_ID}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python scripts/eval_wpu_probe_clean.py \
  --base_model_path "$BASE_MODEL_PATH" \
  --origmodel \
  --lora_config "$LORA_CONFIG" \
  --data_dir "$DATA_DIR" \
  --selected_ids "$SELECTED_IDS" \
  --names_path "$NAMES_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --device "$DEVICE" \
  --seed "$SEED"

echo "Wrote: $OUTPUT_DIR"
