#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run/train_whp.sh 0 1 32 20 1
#   GPU SET_ID LORA_RANK NUM_PASSAGES SEED
#
# Optional debug:
#   SAMPLE_EACH_EPOCH=1 bash run/train_whp.sh 0 1 32 20 1
#   SAMPLE_EACH_EPOCH=1 SAMPLE_MAX_NEW_TOKENS=128 SAMPLE_TEMPERATURE=1.0 bash run/train_whp.sh 0 1 32 20 1
#   SAMPLE_EACH_EPOCH=1 SAMPLE_DO_SAMPLE=1 bash run/train_whp.sh 0 1 32 20 1

GPU_ID="${1:-0}"
SET_ID="${2:-1}"
LORA_RANK="${3:-32}"
NUM_PASSAGES="${4:-20}"
SEED="${5:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
NAMES_PATH="data/whp_names/names_210.json"
SELECTED_IDS="configs/unlearn_ids${SET_ID}.json"
LORA_CONFIG="configs/lora_rank_${LORA_RANK}.json"
OBFUSCATE_PASSAGES="data/whp_samples/set${SET_ID}/obfuscate_samples.json"
OUTPUT_DIR="exp/whp/n${NUM_PASSAGES}_lora${LORA_RANK}_seed${SEED}/set${SET_ID}"
LOGFILE="$OUTPUT_DIR/log.txt"
CONSOLE_LOG="$OUTPUT_DIR/console.log"

BATCH_SIZE=1
LEARNING_RATE=5e-5
NUM_TRAIN_EPOCHS=2
LR_SCHEDULER_TYPE=constant
NUM_WARMUP_RATIO=0.05
LOG_INTERVAL=10
SAVE_INTERVAL=0
MAX_RETRY=5
RETRY_WAIT_SECONDS=1

# Optional epoch-end debug generation.
SAMPLE_EACH_EPOCH="${SAMPLE_EACH_EPOCH:-0}"
SAMPLE_MAX_NEW_TOKENS="${SAMPLE_MAX_NEW_TOKENS:-128}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.0}"
SAMPLE_DO_SAMPLE="${SAMPLE_DO_SAMPLE:-0}"

EXTRA_ARGS=()
if [ "$SAMPLE_EACH_EPOCH" = "1" ]; then
  EXTRA_ARGS+=("--sample_each_epoch")
  EXTRA_ARGS+=("--sample_max_new_tokens" "$SAMPLE_MAX_NEW_TOKENS")
  EXTRA_ARGS+=("--sample_temperature" "$SAMPLE_TEMPERATURE")

  if [ "$SAMPLE_DO_SAMPLE" = "1" ]; then
    EXTRA_ARGS+=("--sample_do_sample")
  fi
fi

for path in "$NAMES_PATH" "$SELECTED_IDS" "$LORA_CONFIG" "$OBFUSCATE_PASSAGES"; do
  if [ ! -f "$path" ]; then
    echo "Missing file: $path"
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
: > "$CONSOLE_LOG"

echo "GPU: $GPU_ID"
echo "Set: $SET_ID"
echo "LoRA rank: $LORA_RANK"
echo "Passages per target: $NUM_PASSAGES"
echo "Seed: $SEED"
echo "Samples: $OBFUSCATE_PASSAGES"
echo "Output: $OUTPUT_DIR"
echo "Sample each epoch: $SAMPLE_EACH_EPOCH"
echo

attempt=1
while true; do
  echo "Starting WHP training, attempt ${attempt}/${MAX_RETRY}"
  echo "===== attempt ${attempt}/${MAX_RETRY} =====" >> "$CONSOLE_LOG"

  if python scripts/train_whp_clean.py \
    --model_path "$MODEL_PATH" \
    --names_path "$NAMES_PATH" \
    --obfuscate_passages "$OBFUSCATE_PASSAGES" \
    --selected_ids "$SELECTED_IDS" \
    --lora_config "$LORA_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --logfile "$LOGFILE" \
    --num_passages "$NUM_PASSAGES" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --num_warmup_ratio "$NUM_WARMUP_RATIO" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$CONSOLE_LOG"; then

    echo "WHP training finished."
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRY" ]; then
    echo "WHP training failed after ${MAX_RETRY} attempts. See $CONSOLE_LOG"
    exit 1
  fi

  echo "Training failed. Retrying in ${RETRY_WAIT_SECONDS} seconds."
  sleep "$RETRY_WAIT_SECONDS"
  attempt=$((attempt + 1))
done

echo "Output directory: $OUTPUT_DIR"
echo "Final checkpoint: $OUTPUT_DIR/checkpoint.final"
echo "Console log: $CONSOLE_LOG"

if [ "$SAMPLE_EACH_EPOCH" = "1" ]; then
  echo "Sample outputs: $OUTPUT_DIR/sample_outputs"
fi