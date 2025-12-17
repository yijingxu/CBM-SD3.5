# Project Map (Results ↔ Code)

This repository contains three independent implementations that produced the results described in the accompanying write-up. Each folder has its own README with run instructions; the pointers below tell you where the code for each set of results lives.

- **SD 3.5 (MM-DiT) experiments → `YijingCode/`**  
  Dual-stream concept bottleneck training and inference hooks for Stable Diffusion 3.5 live here (`train.py`, `inference.py`, supporting modules). See `YijingCode/README.md` for how to prepare the block-12 latent dataset and train/evaluate the SD3.5 concept bottleneck.

- **SD 1.5 experiments → `sd1.5/`**  
  Post-hoc CB-AE for Stable Diffusion 1.5, synthetic data pipeline, and configs. Scripts such as `train_synthetic.py`, `cbae_sd15.py`, and `pseudo_labeler.py` correspond to the SD1.5 results. Usage notes are in `sd1.5/README.md`.

- **DDPM (CelebA-HQ) experiments → `ddpm_cbae/`**  
  Frozen DDPM backbone with mid-block CB-AE, plus evaluation scripts for concept accuracy and interventions. Core entry points: `scripts/train_cbae_ddpm.py`, `scripts/eval_concept_accuracy.py`, `scripts/eval_intervention_ddpm_cs762.py`. Setup and training details are documented in `ddpm_cbae/README.md`.

If you want to reproduce or inspect a particular result, start with the folder above, open its README for run commands, and follow the referenced scripts.
