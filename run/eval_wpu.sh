#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run/eval_wpu.sh 0 1 32 1 exp/whp/n20_lora32_seed1/set1
#   0 = GPU id
#   1 = forget set id
#   2 = LoRA rank
#   3 = checkpoint id, optional. Default 1.
#   4 = run directory

GPU_ID="${1:-0}"
SET_ID="${2:-1}"
LORA_RANK="${3:-32}"
CHECKPOINT_ID="${4:-1}"
RUN_DIR="${5:-exp/unspecified}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATA_DIR="data/whp_probe"
NAMES_PATH="data/whp_names/names_210.json"
SELECTED_IDS="configs/unlearn_ids${SET_ID}.json"
OUTPUT_DIR="${RUN_DIR}/eval"
RUN_NAME="$(basename "$RUN_DIR")_set${SET_ID}"

SEED=1
DEVICE="cuda"
MAX_NEW_TOKENS_OPEN=64
MAX_NEW_TOKENS_MCQ=8
MAX_NEW_TOKENS_YESNO=8
MAX_RETRY=5
RETRY_WAIT_SECONDS=1

# The current configs are named lora_rank_32.json, while older scripts used
# lora_config32.json. Try both, then fall back to a config copied into RUN_DIR.
LORA_CANDIDATES=(
  "configs/lora_rank_${LORA_RANK}.json"
  "configs/lora_config${LORA_RANK}.json"
  "${RUN_DIR}/lora_config.json"
  "${RUN_DIR}/lora_rank_${LORA_RANK}.json"
)
LORA_CONFIG=""
for candidate in "${LORA_CANDIDATES[@]}"; do
  if [ -f "$candidate" ]; then
    LORA_CONFIG="$candidate"
    break
  fi
done
if [ -z "$LORA_CONFIG" ]; then
  echo "Missing LoRA config. Tried:"
  printf '  %s\n' "${LORA_CANDIDATES[@]}"
  exit 1
fi

# Accept both the clean checkpoint directory and older file-style checkpoints.
CHECKPOINT_CANDIDATES=(
  "${RUN_DIR}/checkpoint.${CHECKPOINT_ID}.final"
  "${RUN_DIR}/checkpoint.final"
  "${RUN_DIR}/checkpoint.step${CHECKPOINT_ID}"
  "${RUN_DIR}/checkpoint.${CHECKPOINT_ID}.pt"
  "${RUN_DIR}/checkpoint.0.pt"
  "${RUN_DIR}/pytorch_model.pt"
)
CHECKPOINT=""
for candidate in "${CHECKPOINT_CANDIDATES[@]}"; do
  if [ -d "$candidate" ] && [ -f "$candidate/pytorch_model.pt" ]; then
    CHECKPOINT="$candidate"
    break
  fi
  if [ -f "$candidate" ]; then
    CHECKPOINT="$candidate"
    break
  fi
done
if [ -z "$CHECKPOINT" ]; then
  echo "Missing checkpoint. Tried:"
  printf '  %s\n' "${CHECKPOINT_CANDIDATES[@]}"
  exit 1
fi

for path in "$DATA_DIR" "$SELECTED_IDS" "$NAMES_PATH"; do
  if [ ! -e "$path" ]; then
    echo "Missing path: $path"
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
echo "Run directory: $RUN_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "LoRA config: $LORA_CONFIG"
echo "Data directory: $DATA_DIR"
echo "Names file: $NAMES_PATH"
echo "Selected ids file: $SELECTED_IDS"
echo "Output directory: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo
echo "Selected ids:"
cat "$SELECTED_IDS"
echo

attempt=1
while true; do
  echo
  echo "Starting WPU probe evaluation, attempt ${attempt}/${MAX_RETRY}"

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
    echo "WPU probe evaluation finished successfully."
    echo "Output directory: $OUTPUT_DIR"
    echo "Main summary: $OUTPUT_DIR/summary.csv"
    echo "Open table: $OUTPUT_DIR/brian_table_open.csv"
    echo "MCQ table: $OUTPUT_DIR/brian_table_mcq.csv"
    echo "YesNo table: $OUTPUT_DIR/brian_table_yesno.csv"
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRY" ]; then
    echo
    echo "WPU probe evaluation failed after ${MAX_RETRY} attempts."
    exit 1
  fi

  echo
  echo "Evaluation failed. Retrying in ${RETRY_WAIT_SECONDS} seconds..."
  sleep "$RETRY_WAIT_SECONDS"
  attempt=$((attempt + 1))
done
