#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_generate_whp_teacher_judged.sh 0
#   bash scripts/run_generate_whp_teacher_judged.sh 1
#
# The first argument selects the physical GPU. If omitted, GPU 0 is used.

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
NAMES_PATH="data/origdata/names_210.json"
SELECTED_IDS="config/unlearn_ids.json"
REPLACEMENT_NAMES="data/origdata/names_last_200.json"
OUTPUT_DIR="data/whp_samples"

NUM_SAMPLES=200
SEED=1
MAX_RETRY=5

SOURCE_MAX_NEW_TOKENS=640
SOURCE_TEMPERATURE=0.7
SOURCE_TOP_P=0.9

REWRITE_TEMPERATURE=0.2
REWRITE_TOP_P=0.9
REWRITE_MIN_NEW_TOKENS=480
REWRITE_MAX_NEW_TOKENS_CAP=1024
MAX_REWRITE_ATTEMPTS=3
MAX_SOURCE_REGENERATION_ATTEMPTS=2
JUDGE_MAX_NEW_TOKENS=256

echo "Using GPU: ${GPU_ID}"
echo "Selected IDs file: ${SELECTED_IDS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Target accepted samples: ${NUM_SAMPLES}"

for required_file in "$NAMES_PATH" "$SELECTED_IDS" "$REPLACEMENT_NAMES"; do
  if [ ! -f "$required_file" ]; then
    echo "Missing file: $required_file"
    exit 1
  fi
done

echo "Selected IDs:"
cat "$SELECTED_IDS"
echo

attempt=1
while true; do
  echo
  echo "Starting teacher generation, attempt ${attempt}/${MAX_RETRY}"

  if python scripts/generate_whp_teacher_samples.py \
    --model_path "$MODEL_PATH" \
    --names_path "$NAMES_PATH" \
    --selected_ids "$SELECTED_IDS" \
    --replacement_names_path "$REPLACEMENT_NAMES" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --rewrite_mode llm \
    --resume \
    --do_sample \
    --temperature "$SOURCE_TEMPERATURE" \
    --top_p "$SOURCE_TOP_P" \
    --max_new_tokens "$SOURCE_MAX_NEW_TOKENS" \
    --rewrite_temperature "$REWRITE_TEMPERATURE" \
    --rewrite_top_p "$REWRITE_TOP_P" \
    --rewrite_max_new_tokens 0 \
    --rewrite_min_new_tokens "$REWRITE_MIN_NEW_TOKENS" \
    --rewrite_max_new_tokens_cap "$REWRITE_MAX_NEW_TOKENS_CAP" \
    --max_rewrite_attempts "$MAX_REWRITE_ATTEMPTS" \
    --max_source_regeneration_attempts "$MAX_SOURCE_REGENERATION_ATTEMPTS" \
    --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS"; then

    echo
    echo "Teacher generation finished."
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRY" ]; then
    echo
    echo "Teacher generation failed after ${MAX_RETRY} attempts."
    exit 1
  fi

  echo
  echo "Generation failed. Retrying in 10 seconds."
  sleep 10
  attempt=$((attempt + 1))
done
