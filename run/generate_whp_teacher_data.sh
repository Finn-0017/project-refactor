#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Basic settings
# ============================================================

# Usage：
#   bash generate_whp_teacher_data.sh 0
#   bash generate_whp_teacher_data.sh 1
#   bash generate_whp_teacher_data.sh 2
#   bash generate_whp_teacher_data.sh 3

GPU_ID="${1:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
NAMES_PATH="data/origdata/names_210.json"
SELECTED_IDS="configs/unlearn_ids1.json"
REPLACEMENT_NAMES="data/origdata/names_last_200.json"
OUTPUT_DIR="data/whp_samples"

NUM_SAMPLES=200
SEED=1

# ============================================================
# Generation settings
# ============================================================

TEMPERATURE=0.7
TOP_P=0.9
MAX_NEW_TOKENS=320
MAX_RETRY=5

echo "Using physical GPU: ${GPU_ID}"
echo "Selected ids file: ${SELECTED_IDS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Target samples per person: ${NUM_SAMPLES}"

# ============================================================
# Run with resume
# ============================================================

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
    --do_sample \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
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
  echo "Generation failed. Retrying in 10 seconds..."
  sleep 10
  attempt=$((attempt + 1))
done