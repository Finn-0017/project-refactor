#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
SET_ID="1"
SEED="1"
NUM_SAMPLES="20"
OUTDIR="exp/teacher_samples_set${SET_ID}_n${NUM_SAMPLES}_seed${SEED}"

python scripts/generate_whp_teacher_samples.py \
  --model_path "$MODEL_PATH" \
  --names_path data/WHPplus/whp_names.json \
  --selected_ids config/unlearn_ids${SET_ID}.json \
  --output_dir "$OUTDIR" \
  --num_samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --rewrite_mode llm \
  --do_sample

# The output can then be used by WHP training:
# python scripts/train_whp_clean.py \
#   --model_path "$MODEL_PATH" \
#   --names_path data/WHPplus/whp_names.json \
#   --obfuscate_passages "$OUTDIR/obfuscate_samples.json" \
#   --selected_ids config/unlearn_ids${SET_ID}.json \
#   --num_passages "$NUM_SAMPLES" \
#   --lora_config config/lora_config10.json \
#   --output_dir exp/clean_whp_generated_set${SET_ID}_n${NUM_SAMPLES}_seed${SEED}
