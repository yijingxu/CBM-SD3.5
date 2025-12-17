# Replicating DDPM Concept Bottleneck Results

This README provides instructions for replicating the DDPM concept bottleneck experiments from the paper.

## Overview

We train a Concept Bottleneck Autoencoder (CB-AE) on a frozen pretrained DDPM (`google/ddpm-celebahq-256`) to learn interpretable concept representations for four binary facial attributes: **Smiling**, **Young**, **Male**, and **Eyeglasses**.

## Requirements

```bash
pip install torch torchvision diffusers datasets transformers huggingface_hub
pip install clip  # For CLIP-based pseudo-labeling
```

## Directory Structure

```
ddpm_cbae/
├── scripts/
│   ├── train_cbae_ddpm.py          # Training script
│   ├── eval_intervention_ddpm_cs762.py  # Evaluation script
│   └── run_all_evals.py            # Batch evaluation script
├── models/
│   ├── cbae_ddpm.py                # CB-AE + DDPM wrapper
│   ├── cbae_unet2d.py              # CB-AE architecture & modified UNet
│   └── clip_pseudolabeler.py       # CLIP-based pseudo-labeling
├── checkpoints_local/              # Trained CB-AE checkpoints (created during training)
│   ├── cbae_ddpm_step1000.pt
│   ├── cbae_ddpm_step2000.pt
│   └── ...
├── results/                        # Evaluation results (created during eval)
│   ├── cbae_ddpm_step10000/
│   │   ├── eval_Smiling_target1/
│   │   │   ├── 000_orig.png
│   │   │   ├── 000_interv.png
│   │   │   └── ...
│   │   ├── eval_Young_target1/
│   │   ├── eval_Male_target1/
│   │   └── eval_Eyeglasses_target1/
│   └── eval_summary.txt
└── checkpoints/                    # Final checkpoint
    └── cbae_ddpm_final.pt
```

---

## Step 1: Training the CB-AE

Train the CB-AE for 10,000 steps with the following command:

```bash
python scripts/train_cbae_ddpm.py \
    --dataset_id flwrlabs/celeba \
    --split train \
    --batch_size 16 \
    --total_steps 10000 \
    --log_every 200 \
    --save_every 1000 \
    --pretrained_model_id google/ddpm-celebahq-256 \
    --max_timestep 400 \
    --hidden_dim 1024 \
    --lr 1e-4 \
    --w_lr1 1.0 \
    --w_lr2 1.0 \
    --w_lc 1.0 \
    --w_li1 1.0 \
    --w_li2 1.0 \
    --p_intervene 1.0 \
    --intervene_idx 0 \
    --seed 123
```

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--batch_size` | 16 | Training batch size |
| `--total_steps` | 10000 | Total training iterations |
| `--max_timestep` | 400 | CB-AE active for t ≤ 400 |
| `--hidden_dim` | 1024 | CB-AE hidden layer dimension |
| `--lr` | 1e-4 | Learning rate (Adam) |
| `--w_lr1` | 1.0 | Latent reconstruction loss weight |
| `--w_lr2` | 1.0 | Output preservation loss weight |
| `--w_lc` | 1.0 | Concept alignment loss weight |
| `--w_li1` | 1.0 | Intervention loss (CLIP) weight |
| `--w_li2` | 1.0 | Intervention loss (cycle) weight |
| `--p_intervene` | 1.0 | Intervention probability per batch |

### Output

- **Checkpoints**: Saved to `checkpoints_local/cbae_ddpm_step{N}.pt` every 1000 steps
- **Final checkpoint**: Saved to `checkpoints/cbae_ddpm_final.pt`

### Resuming Training

To resume from a checkpoint:

```bash
python scripts/train_cbae_ddpm.py \
    --dataset_id flwrlabs/celeba \
    --split train \
    --batch_size 16 \
    --total_steps 10000 \
    --log_every 200 \
    --save_every 1000 \
    --pretrained_model_id google/ddpm-celebahq-256 \
    --max_timestep 400 \
    --hidden_dim 1024 \
    --lr 1e-4 \
    --w_lr1 1.0 \
    --w_lr2 1.0 \
    --w_lc 1.0 \
    --w_li1 1.0 \
    --w_li2 1.0 \
    --p_intervene 1.0 \
    --intervene_idx 0 \
    --seed 123 \
    --resume_from_local checkpoints_local/cbae_ddpm_step5000.pt \
    --resume_optimizer
```

---

## Step 2: Evaluating Concept Interventions

### Single Evaluation

To evaluate intervention on a single concept (e.g., Smiling → target=1):

```bash
python -m scripts.eval_intervention_ddpm_cs762 \
    --ckpt checkpoints_local/cbae_ddpm_step10000.pt \
    --pretrained_model_id google/ddpm-celebahq-256 \
    --max_timestep 400 \
    --hidden_dim 1024 \
    --concept_idx 0 \
    --target 1 \
    --num_batches 5 \
    --batch_size 16 \
    --ddim_steps 50 \
    --outdir results/eval_Smiling_target1 \
    --seed 0
```

### Concept Index Mapping

| Concept | Index | Target=0 | Target=1 |
|---------|-------|----------|----------|
| Smiling | 0 | Not Smiling | Smiling |
| Young | 1 | Old | Young |
| Male | 2 | Female | Male |
| Eyeglasses | 3 | No Glasses | Glasses |

### Evaluation Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num_batches` | 5 | Number of batches to evaluate |
| `--batch_size` | 16 | Images per batch (80 total) |
| `--ddim_steps` | 50 | DDIM sampling steps |
| `--save_batches` | 5 | Number of batches to save as images |

### Batch Evaluation (All Concepts)

To evaluate all concepts across all checkpoints:

```bash
python scripts/run_all_evals.py \
    --ckpt_dir checkpoints_local \
    --results_dir results \
    --batch_size 16 \
    --num_batches 5 \
    --ddim_steps 50 \
    --targets 1 \
    --seed 0
```

To evaluate specific concepts only:

```bash
python scripts/run_all_evals.py \
    --ckpt_dir checkpoints_local \
    --results_dir results \
    --batch_size 16 \
    --num_batches 5 \
    --ddim_steps 50 \
    --targets 0 1 \
    --concepts Smiling Eyeglasses \
    --seed 0
```

---

## Step 3: Output Locations

### Trained Models

| File | Location | Description |
|------|----------|-------------|
| Intermediate checkpoints | `checkpoints_local/cbae_ddpm_step{N}.pt` | Saved every 1000 steps |
| Final checkpoint | `checkpoints/cbae_ddpm_final.pt` | Saved at end of training |

### Intervention Results

| Output | Location | Description |
|--------|----------|-------------|
| Original images | `results/{ckpt_name}/eval_{concept}_target{T}/{batch}_orig.png` | Generated without intervention |
| Intervened images | `results/{ckpt_name}/eval_{concept}_target{T}/{batch}_interv.png` | Generated with concept intervention |
| Evaluation summary | `results/eval_summary.txt` | Summary of all evaluation runs |

Example paths:
```
results/cbae_ddpm_step10000/eval_Smiling_target1/000_orig.png
results/cbae_ddpm_step10000/eval_Smiling_target1/000_interv.png
results/cbae_ddpm_step10000/eval_Eyeglasses_target1/000_orig.png
results/cbae_ddpm_step10000/eval_Eyeglasses_target1/000_interv.png
```

