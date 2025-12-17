# SD 3.5 (MM-DiT) Concept Bottleneck

This folder contains the Stable Diffusion 3.5 Medium concept bottleneck implementation that produced the SD3.5 results.

## Where the code lives
- Dual-stream CB-AE architecture: `CBmodel.py`
- Training loop and checkpointing: `train.py`
- Latent dataset specification: `training_data_design.md` (block-12 latent/text/noise tensors)
- Inference-time intervention hook: `inference.py`

## How to run
1. Prepare block-12 latent data as described in `training_data_design.md` (defaults expect `YijingCode/TrainingData/`).
2. Train the CB-AE:
   ```bash
   python YijingCode/train.py --data_dir YijingCode/TrainingData --checkpoint_dir YijingCode/checkpoints
   ```
   (Adjust hyperparameters via CLI flags; they are saved to `training_config.json` alongside checkpoints.)
3. Run inference with concept forcing using a trained checkpoint:
   ```bash
   python YijingCode/inference.py \
     --prompt "a bowl of pasta" \
     --config YijingCode/checkpoints/training_config.json \
     --checkpoint YijingCode/checkpoints/best_model.pt \
     --concept_a Spaghetti --concept_b Plate \
     --intervention_start_t 6 --intervention_end_t 12
   ```
   The CLI lets you pick concept targets, the intervention window, the hooked transformer block, CFG scale, and output path.
