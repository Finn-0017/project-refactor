#!/usr/bin/env bash
set -euo pipefail

# Evaluate one trained checkpoint on the current data/whp_probe files.
# Edit these variables or override them from the command line, for example:
#   SET_ID=3 RUN_DIR=exp/my_run LORA_CONFIG=config/lora_config10.json bash scripts/eval_wpu_probe_one_set.sh

SET_ID="${SET_ID:-1}"
RUN_DIR="${RUN_DIR:-exp/your_run_dir}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-data/whp_probe}"
NAMES_PATH="${NAMES_PATH:-data/WHPplus/whp_names.json}"
SELECTED_IDS="${SELECTED_IDS:-config/unlearn_ids${SET_ID}.json}"
LORA_CONFIG="${LORA_CONFIG:-config/lora_config10.json}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoint.1.final}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_wpu_probe_set${SET_ID}}"
RUN_NAME="${RUN_NAME:-$(basename "${RUN_DIR}")_set${SET_ID}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python scripts/eval_wpu_probe_clean.py \
  --base_model_path "$BASE_MODEL_PATH" \
  --run_dir "$RUN_DIR" \
  --checkpoint "$CHECKPOINT" \
  --lora_config "$LORA_CONFIG" \
  --data_dir "$DATA_DIR" \
  --selected_ids "$SELECTED_IDS" \
  --names_path "$NAMES_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --device "$DEVICE" \
  --seed "$SEED"

echo "Wrote: $OUTPUT_DIR"
echo "Main summary: $OUTPUT_DIR/summary.csv"
echo "Open table:   $OUTPUT_DIR/brian_table_open.csv"
echo "MCQ table:    $OUTPUT_DIR/brian_table_mcq.csv"
echo "YesNo table:  $OUTPUT_DIR/brian_table_yesno.csv"
