#!/usr/bin/env bash
set -euo pipefail

# Usage：
#   bash generate_whp_teacher_data.sh 0
#   bash generate_whp_teacher_data.sh 1
#   bash generate_whp_teacher_data.sh 2
#   bash generate_whp_teacher_data.sh 3

GPU_ID="${1:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# ============================================================
# Files
# ============================================================

MODEL_PATH="/rds/user/xy319/hpc-work/projects/project-coding/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
NAMES_PATH="data/origdata/names_210.json"
SELECTED_IDS="configs/unlearn_ids1.json"
REPLACEMENT_NAMES="data/origdata/names_last_200.json"
OUTPUT_DIR="data/whp_samples/set1"

# ============================================================
# Experiment settings
# ============================================================

NUM_SAMPLES=200
SEED=1
MAX_RETRY=5

# Step 1: generate a source biography about the unrelated person.
SOURCE_MAX_NEW_TOKENS=400
SOURCE_TEMPERATURE=0.7
SOURCE_TOP_P=0.9

# Step 2: rewrite the source biography into a fake biography about the target.
REWRITE_MAX_NEW_TOKENS=400
REWRITE_TEMPERATURE=0.2
REWRITE_TOP_P=0.95

# Step 3.1: hard length check.
TARGET_WORDS=300
MIN_WORDS=220
MAX_WORDS=380

# Step 3.2: LLM judge. Rejected rewrites are sent back to Step 2.
MAX_REWRITE_ATTEMPTS=4
JUDGE_MAX_NEW_TOKENS=256

# Source passage is generated once. This only retries if the source generation is
# empty/refusal-like before rewrite starts.
MAX_SOURCE_ATTEMPTS=2

printf 'Using physical GPU: %s\n' "$GPU_ID"
printf 'Selected ids file: %s\n' "$SELECTED_IDS"
printf 'Output directory: %s\n' "$OUTPUT_DIR"
printf 'Target samples per person: %s\n' "$NUM_SAMPLES"

for required_file in "$NAMES_PATH" "$SELECTED_IDS" "$REPLACEMENT_NAMES"; do
  if [ ! -f "$required_file" ]; then
    echo "Missing file: $required_file"
    exit 1
  fi
done

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
    --rewrite_mode llm \
    --resume \
    --do_sample \
    --source_max_new_tokens "$SOURCE_MAX_NEW_TOKENS" \
    --temperature "$SOURCE_TEMPERATURE" \
    --top_p "$SOURCE_TOP_P" \
    --rewrite_max_new_tokens "$REWRITE_MAX_NEW_TOKENS" \
    --rewrite_temperature "$REWRITE_TEMPERATURE" \
    --rewrite_top_p "$REWRITE_TOP_P" \
    --target_words "$TARGET_WORDS" \
    --min_words "$MIN_WORDS" \
    --max_words "$MAX_WORDS" \
    --max_rewrite_attempts "$MAX_REWRITE_ATTEMPTS" \
    --max_source_attempts "$MAX_SOURCE_ATTEMPTS" \
    --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS"; then

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
