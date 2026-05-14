#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
NAMES_PATH="${NAMES_PATH:-data/WHPplus/whp_names_210.json}"
SELECTED_IDS="${SELECTED_IDS:-config/unlearn_ids_first10.json}"
REPLACEMENT_NAMES="${REPLACEMENT_NAMES:-data/WHPplus/whp_names_210_replacement_names_200.json}"
OUTDIR="${OUTDIR:-exp/teacher_samples_first10_source200_seed1}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
SEED="${SEED:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"

mkdir -p "$OUTDIR"

while true; do
  echo "[teacher generation] starting or resuming in $OUTDIR"
  if python scripts/generate_whp_teacher_samples.py \
      --model_path "$MODEL_PATH" \
      --names_path "$NAMES_PATH" \
      --selected_ids "$SELECTED_IDS" \
      --replacement_names_path "$REPLACEMENT_NAMES" \
      --output_dir "$OUTDIR" \
      --num_samples "$NUM_SAMPLES" \
      --seed "$SEED" \
      --rewrite_mode llm \
      --resume \
      --do_sample \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --max_new_tokens "$MAX_NEW_TOKENS"; then
    echo "[teacher generation] completed"
    break
  fi

  echo "[teacher generation] interrupted or failed; retrying in ${RETRY_SLEEP_SECONDS}s" >&2
  sleep "$RETRY_SLEEP_SECONDS"
done
