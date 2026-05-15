#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run/eval_wpu_orig_model.sh 0 1
#   0 = GPU id
#   1 = forget set id

GPU_ID="${1:-0}"
SET_ID="${2:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

DATA_DIR="data/whp_probe"
NAMES_PATH="data/whp_names/names_210.json"
SELECTED_IDS="configs/unlearn_ids${SET_ID}.json"
OUTPUT_DIR="exp/orig_model_eval/set${SET_ID}"
RUN_NAME="initial_model_set${SET_ID}"
LORA_CONFIG="configs/lora_rank_256.json"

SEED=1
DEVICE="cuda"
MAX_NEW_TOKENS_OPEN=64
MAX_NEW_TOKENS_MCQ=32
MAX_NEW_TOKENS_YESNO=8

MAX_RETRY=5
RETRY_WAIT_SECONDS=1

if [ ! -d "$DATA_DIR" ]; then
  echo "Missing data directory: $DATA_DIR"
  exit 1
fi

if [ ! -f "$SELECTED_IDS" ]; then
  echo "Missing selected ids file: $SELECTED_IDS"
  exit 1
fi

if [ ! -f "$NAMES_PATH" ]; then
  echo "Missing names file: $NAMES_PATH"
  exit 1
fi

if [ ! -f "$LORA_CONFIG" ]; then
  echo "Missing LoRA config file: $LORA_CONFIG"
  exit 1
fi

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
  if [ ! -f "$DATA_DIR/$file_name" ]; then
    echo "Missing probe file: $DATA_DIR/$file_name"
    exit 1
  fi
done

echo "Using physical GPU: $GPU_ID"
echo "Model path: $MODEL_PATH"
echo "Data directory: $DATA_DIR"
echo "Names file: $NAMES_PATH"
echo "Selected ids file: $SELECTED_IDS"
echo "Output directory: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo "MCQ scoring: raw full-vocabulary choice probabilities, argmax over A/B/C/D/E"
echo
echo "Selected ids:"
cat "$SELECTED_IDS"
echo

attempt=1

mkdir -p "$OUTPUT_DIR"
EVAL_LOG="$OUTPUT_DIR/eval_console.log"
: > "$EVAL_LOG"

while true; do
  echo
  echo "Starting initial model WPU probe evaluation, attempt ${attempt}/${MAX_RETRY}"

  if python scripts/eval_wpu_probe_clean.py \
    --base_model_path "$MODEL_PATH" \
    --origmodel \
    --lora_config "$LORA_CONFIG" \
    --data_dir "$DATA_DIR" \
    --selected_ids "$SELECTED_IDS" \
    --names_path "$NAMES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --max_new_tokens_open "$MAX_NEW_TOKENS_OPEN" \
    --max_new_tokens_mcq "$MAX_NEW_TOKENS_MCQ" \
    --max_new_tokens_yesno "$MAX_NEW_TOKENS_YESNO" \
    >> "$EVAL_LOG" 2>&1; then

    echo
    echo "Initial model WPU probe evaluation finished successfully."
    echo "Output directory: $OUTPUT_DIR"
    echo "Main summary: $OUTPUT_DIR/summary.csv"
    echo "Open table: $OUTPUT_DIR/brian_table_open.csv"
    echo "MCQ table: $OUTPUT_DIR/brian_table_mcq.csv"
    echo "YesNo table: $OUTPUT_DIR/brian_table_yesno.csv"
    echo "Console log: $EVAL_LOG"
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRY" ]; then
    echo
    echo "Initial model WPU probe evaluation failed after ${MAX_RETRY} attempts."
    exit 1
  fi

  echo
  echo "Evaluation failed. Retrying in ${RETRY_WAIT_SECONDS} seconds..."
  sleep "$RETRY_WAIT_SECONDS"
  attempt=$((attempt + 1))
done
