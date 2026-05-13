#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/train_df_mcq_clean.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --train_data_path data/WHPplus/balanced_whp_mcq_train_dedup.json \
  --selected_ids config/unlearn_ids1.json \
  --lora_config config/lora_config10.json \
  --output_dir exp/clean_dfmcq_set1_lora10_seed1 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --retain_factor 1.0 \
  --seed 1
