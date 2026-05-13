# Refactor notes

## What was cleaned

1. **Training logic is separated from data loading.**
   - `data.py` only builds datasets and records which samples are used.
   - `trainer.py` only computes losses and updates model parameters.
   - `modeling.py` only loads the model, attaches LoRA, and saves checkpoints.

2. **MCQ entropy is computed on a normalized choice distribution.**
   - `choice.py` extracts logits for A/B/C/D/E.
   - Softmax is applied over the choice logits only.
   - Entropy is computed from this normalized distribution.

3. **WHP passage selection is reproducible.**
   - `nested_indices(...)` shuffles each person's pool once for a fixed seed.
   - Smaller data sizes are prefixes of larger data sizes.
   - The selected passage IDs are written to `data_manifest.json`.

4. **Experiment metadata are saved by default.**
   - `run_config.json` records the command-line configuration.
   - `data_manifest.json` records file hashes and selected samples.
   - `parameter_summary.json` records the number of trainable parameters.

5. **Teacher generation is explicit instead of hidden inside training.**
   - `generate_whp_teacher_samples.py` creates WHP obfuscation passages before training.
   - `train_whp_clean.py` only consumes a passage JSON file and optimizes the model.
   - This makes it possible to reuse the exact same teacher samples across LoRA-rank and optimizer sweeps.

6. **The public code path avoids hidden branches.**
   - The clean scripts focus on DF-MCQ, WHP with precomputed/generated passages, and evaluation.
   - Older exploratory branches such as SelfCheck and GRPO are not removed from the original repository; they are simply not part of the clean path.

## Why this helps the next experiment

The next experiment needs to distinguish data amount from data identity. The WHP script now makes this explicit by saving selected passage IDs. When comparing `n=20` and `n=50`, the run metadata can verify that the `n=20` condition is a subset of the `n=50` condition.

The next experiment also needs to distinguish method effects from LoRA capacity. The parameter summary and copied LoRA config make it easy to check that rank, alpha, target modules, and trainable-parameter counts are constant within a data-amount sweep.

## Suggested integration path

1. Copy `src/unlearning_research/` and the clean scripts into the repository root.
2. Run one dry run on a tiny selected subset or a small local model to check paths.
3. Generate WHP teacher samples once with `generate_whp_teacher_samples.py`.
4. Train WHP with `train_whp_clean.py --obfuscate_passages <teacher_dir>/obfuscate_samples.json`.
5. Re-run the original baseline with the clean script and compare refusal/MCQ metrics.
6. Use the same generated teacher pool for the data-amount sweep so data identity is controlled.
7. Keep the original scripts until results match within expected stochastic variation.
