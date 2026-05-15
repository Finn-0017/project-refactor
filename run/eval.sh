#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run/eval.sh 0 32 20 1
#   GPU LORA_RANK NUM_PASSAGES SEED

GPU_ID="${1:-0}"
LORA_RANK="${2:-32}"
NUM_PASSAGES="${3:-20}"
SEED="${4:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

DATA_DIR="data/whp_probe"
NAMES_PATH="data/whp_names/names_210.json"
LORA_CONFIG="configs/lora_rank_${LORA_RANK}.json"

DEVICE="cuda"
MAX_NEW_TOKENS_OPEN=64
MAX_NEW_TOKENS_MCQ=8
MAX_NEW_TOKENS_YESNO=8
MAX_RETRY=5
RETRY_WAIT_SECONDS=1

if [ ! -d "$DATA_DIR" ]; then
  echo "Missing data directory: $DATA_DIR"
  exit 1
fi

for path in "$LORA_CONFIG" "$NAMES_PATH"; do
  if [ ! -f "$path" ]; then
    echo "Missing file: $path"
    exit 1
  fi
done

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
echo "LoRA config: $LORA_CONFIG"
echo "Data directory: $DATA_DIR"
echo "Names file: $NAMES_PATH"
echo "Passages per target: $NUM_PASSAGES"
echo "Seed: $SEED"
echo

for SET_ID in 1 2 3 4 5; do
  SELECTED_IDS="configs/unlearn_ids${SET_ID}.json"
  RUN_DIR="exp/whp/n${NUM_PASSAGES}_lora${LORA_RANK}_seed${SEED}/set${SET_ID}"
  CHECKPOINT="${RUN_DIR}/checkpoint.final"
  OUTPUT_DIR="${RUN_DIR}/eval"
  RUN_NAME="whp_n${NUM_PASSAGES}_lora${LORA_RANK}_seed${SEED}_set${SET_ID}"

  if [ ! -f "$SELECTED_IDS" ]; then
    echo "Missing selected ids file: $SELECTED_IDS"
    exit 1
  fi

  if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint directory not found: $CHECKPOINT"
    exit 1
  fi

  if [ ! -f "$CHECKPOINT/pytorch_model.pt" ]; then
    echo "Checkpoint weight file not found: $CHECKPOINT/pytorch_model.pt"
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"

  echo "============================================================"
  echo "Starting evaluation for set${SET_ID}"
  echo "Run directory: $RUN_DIR"
  echo "Checkpoint: $CHECKPOINT"
  echo "Selected ids file: $SELECTED_IDS"
  echo "Output directory: $OUTPUT_DIR"
  echo "Run name: $RUN_NAME"
  echo "Selected ids:"
  cat "$SELECTED_IDS"
  echo
  echo "============================================================"

  attempt=1

  while true; do
    echo
    echo "Starting WPU probe evaluation for set${SET_ID}, attempt ${attempt}/${MAX_RETRY}"

    if python scripts/eval_wpu_probe_clean.py \
      --base_model_path "$MODEL_PATH" \
      --run_dir "$RUN_DIR" \
      --checkpoint "$CHECKPOINT" \
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
      --max_new_tokens_yesno "$MAX_NEW_TOKENS_YESNO"; then

      echo
      echo "set${SET_ID} evaluation finished."
      echo "Output directory: $OUTPUT_DIR"
      echo "Main summary: $OUTPUT_DIR/summary.csv"
      echo "Open table: $OUTPUT_DIR/brian_table_open.csv"
      echo "MCQ table: $OUTPUT_DIR/brian_table_mcq.csv"
      echo "YesNo table: $OUTPUT_DIR/brian_table_yesno.csv"
      break
    fi

    if [ "$attempt" -ge "$MAX_RETRY" ]; then
      echo
      echo "set${SET_ID} evaluation failed after ${MAX_RETRY} attempts."
      exit 1
    fi

    echo
    echo "set${SET_ID} evaluation failed. Retrying in ${RETRY_WAIT_SECONDS} seconds..."
    sleep "$RETRY_WAIT_SECONDS"
    attempt=$((attempt + 1))
  done

  echo
done

echo "All set evaluations finished."
echo "Base output directory: exp/whp/n${NUM_PASSAGES}_lora${LORA_RANK}_seed${SEED}"