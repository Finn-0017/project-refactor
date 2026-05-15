#!/usr/bin/env bash
set -euo pipefail

# Usage：
#   bash generate_teacher.sh 0 1
#   0 = GPU id
#   1 = forget set id

GPU_ID="${1:-0}"
SET_ID="${2:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

NAMES_PATH="data/whp_names/names_210.json"
SELECTED_IDS="configs/unlearn_ids${SET_ID}.json"
REPLACEMENT_NAMES="data/whp_names/names_last_200.json"
OUTPUT_DIR="data/whp_samples/set${SET_ID}"

NUM_SAMPLES=200
SEED=1

TEMPERATURE=0.7
TOP_P=0.9
MAX_NEW_TOKENS=320
SENTENCE_COMPLETION_EXTRA_TOKENS=96

MAX_RETRY=5
RETRY_WAIT_SECONDS=1

if [ ! -f "$NAMES_PATH" ]; then
  echo "Missing names file: $NAMES_PATH"
  exit 1
fi

if [ ! -f "$SELECTED_IDS" ]; then
  echo "Missing selected ids file: $SELECTED_IDS"
  exit 1
fi

if [ ! -f "$REPLACEMENT_NAMES" ]; then
  echo "Missing replacement names file: $REPLACEMENT_NAMES"
  exit 1
fi

echo "Using physical GPU: $GPU_ID"
echo "Model path: $MODEL_PATH"
echo "Names file: $NAMES_PATH"
echo "Selected ids file: $SELECTED_IDS"
echo "Replacement names file: $REPLACEMENT_NAMES"
echo "Output directory: $OUTPUT_DIR"
echo "Samples per target: $NUM_SAMPLES"
echo
echo "Selected ids:"
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
    --rewrite_mode string \
    --do_sample \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --sentence_completion_extra_tokens "$SENTENCE_COMPLETION_EXTRA_TOKENS" \
    --resume; then

    echo
    echo "Teacher generation finished successfully."
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRY" ]; then
    echo
    echo "Teacher generation failed after ${MAX_RETRY} attempts."
    exit 1
  fi

  echo
  echo "Generation failed. Retrying in ${RETRY_WAIT_SECONDS} seconds..."
  sleep "$RETRY_WAIT_SECONDS"
  attempt=$((attempt + 1))
done