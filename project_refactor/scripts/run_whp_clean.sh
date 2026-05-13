#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/train_whp_clean.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --names_path data/WHPplus/whp_names.json \
  --obfuscate_passages data/WHPplus/all_obfuscate_samples.json \
  --selected_ids config/unlearn_ids1.json \
  --num_passages 20 \
  --lora_config config/lora_config10.json \
  --output_dir exp/clean_whp_set1_n20_lora10_seed1 \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --lr_scheduler_type constant \
  --seed 1
