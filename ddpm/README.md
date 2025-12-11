# CB-AE DDPM Training

Complete training setup for Concept Bottleneck Autoencoder (CB-AE) with DDPM models.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Training Pipeline](#training-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository contains code to train a **Concept Bottleneck Autoencoder (CB-AE)** integrated with pre-trained DDPM (Denoising Diffusion Probabilistic Model) for controllable image generation.

### Key Features:
- **Modular CB-AE architecture** with encoder and decoder
- **Support for real labels** from annotation files (recommended)
- **Support for pseudo-labels** using CLIP zero-shot classification
- **Multi-concept training** (supports 2-10 binary concepts)
- **Intervention capability** for controllable generation
- **Early timestep training** (t ≤ 400) where latents are less noisy

### Model Architecture:
```
DDPM UNet → Latent w_t → CB-AE Encoder → Concepts → CB-AE Decoder → Reconstructed Latent → Noise Prediction
```

---

## Directory Structure

```
/data/jmu27/posthocbm4/
├── config/
│   └── cbae_ddpm/
│       └── celebahq_full.yaml         # Training configuration
├── models/
│   ├── cbae_ddpm.py                   # DDPM-specific CB-AE wrapper
│   ├── cbae_core.py                   # Core CB-AE module (backend-agnostic)
│   ├── cbae_unet2d.py                 # Modified UNet2D with latent extraction
│   ├── basic.py                       # Base model class
│   ├── clip_pseudolabeler.py          # CLIP pseudo-labeling utilities
│   └── checkpoints/                   # Model checkpoints saved here
│       └── celebahq_cbae_ddpm_full_4775imgs_8concepts_cbae.pt
├── train/
│   ├── train_cbae_ddpm.py             # Training with pseudo-labels (CLIP)
│   └── train_cbae_ddpm_custom.py      # Training with real labels (recommended)
└── utils/
    └── utils.py                       # Dataset loaders and utilities
```

---

## Installation

### Requirements:
```bash
# Core dependencies
pip install torch torchvision
pip install diffusers accelerate transformers
pip install tqdm PyYAML tensorboard pillow
```

### Verify Installation:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import diffusers; print(diffusers.__version__)"
```

---

## Data Preparation

### Option 1: Using Real Labels (Recommended)

**Annotation File Format:**
```
<num_images>
<concept_1> <concept_2> ... <concept_n>
image001.jpg 0 1 0 1 1 0 1 0
image002.jpg 1 0 1 0 0 1 0 1
...
```

**Example** (`train_balanced.txt`):
```
4775
attractive lipstick mouth-closed smiling cheekbones makeup gender eyebrows
000001.jpg 1 0 0 1 0 1 0 1
000002.jpg 0 1 1 0 1 0 1 0
...
```

- Line 1: Total number of images
- Line 2: Concept names (space-separated)
- Lines 3+: `filename.jpg label1 label2 ... labelN`
- Labels: `0` or `1` for binary concepts

**Directory Structure:**
```
/data/jmu27/posthoc-generative-cbm/datasets/CelebA-HQ/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── train_balanced.txt
└── test_balanced.txt
```

### Option 2: Using Pseudo-Labels (CLIP)

If you don't have annotations, the model can use CLIP to generate pseudo-labels automatically. Just set `use_real_labels: false` in the config.

---

## Configuration

### Main Config File: `config/cbae_ddpm/celebahq_full.yaml`

#### Dataset Section:
```yaml
dataset:
  name: celebahq                    # Dataset name
  img_size: 256                     # Image resolution
  batch_size: 4                     # Batch size (adjust for GPU memory)
  num_workers: 4                    # DataLoader workers
  data_path: /path/to/CelebA-HQ     # Root directory with images/

  # For real labels (recommended):
  use_real_labels: true
  train_anno: /path/to/train_balanced.txt
  test_anno: /path/to/test_balanced.txt
```

#### Model Section:
```yaml
model:
  pretrained: google/ddpm-celebahq-256   # HuggingFace DDPM model
  type: cbae_ddpm
  latent_shape: [512, 8, 8]              # UNet mid-block latent shape
  latent_noise_dim: 512                  # Hidden dimension for CB-AE
  max_timestep: 400                      # Apply CB-AE for t ≤ 400

  concepts:
    concept_bins: [2, 2, 2, 2, 2, 2, 2, 2]    # Number of classes per concept
    concept_names: [                          # Concept names
      "attractive", "lipstick", "mouth-closed",
      "smiling", "cheekbones", "makeup",
      "gender", "eyebrows"
    ]
    emb_size: 128                             # CB-AE embedding dimension
    concept_output: [2, 2, 2, 2, 2, 2, 2, 2]  # Output dims (same as bins)
    types: ["bin", "bin", "bin", ...]         # All binary
```

**Important Parameters:**
- `max_timestep: 400` - CB-AE only processes timesteps ≤400 (less noisy latents)
- `emb_size: 128` - Larger = better capacity, but more memory
- `concept_bins` - Number of values per concept (use `[2, 2, ...]` for binary)

#### Training Section:
```yaml
train_config:
  epochs: 50                   # Total epochs
  recon_lr: 0.0002            # Learning rate for reconstruction
  conc_lr: 0.0002             # Learning rate for intervention
  betas: [0.5, 0.99]          # Adam optimizer betas
  save_model: true            # Save checkpoints
  use_cuda: true              # Use GPU
  log_interval: 100           # Log every N steps
  steps_per_epoch: 1193       # Limit steps per epoch (optional)
  pl_prob_thresh: 0           # Pseudo-label threshold (0 = no filtering)
  plot_loss: true             # Plot losses to TensorBoard
```

---

## Training

### Quick Start (Using Existing Config):

```bash
cd /data/jmu27/posthocbm4

# Train with real labels (recommended)
python train/train_cbae_ddpm_custom.py \
    -d celebahq \
    -e cbae_ddpm \
    -t my_experiment_name \
    -p clipzs
```

### Command-Line Arguments:

```bash
python train/train_cbae_ddpm_custom.py \
    -d <dataset>              # Dataset: celebahq, celeba64, cub64
    -e <expt-name>            # Experiment name (folder for checkpoints)
    -t <tensorboard-name>     # TensorBoard run name
    -p <pseudo-label>         # Pseudo-label method: clipzs, supervised, tipzs
    [--load-pretrained]       # Resume from checkpoint
    [--pretrained-load-name <filename>]  # Checkpoint filename to load
```

### Training Examples:

#### 1. Train from Scratch (Real Labels):
```bash
python train/train_cbae_ddpm_custom.py \
    -d celebahq \
    -e cbae_ddpm \
    -t full_8concepts_50epochs \
    -p clipzs
```

**What happens:**
- Loads config from `config/cbae_ddpm/celebahq_full.yaml`
- Uses real labels from `train_balanced.txt` and `test_balanced.txt`
- Trains for 50 epochs
- Saves checkpoints to `models/checkpoints/celebahq_cbae_ddpm_full_8concepts_50epochs_cbae.pt`
- Logs to TensorBoard: `runs/celebahq_cbae_ddpm_full_8concepts_50epochs/`

#### 2. Resume Training from Checkpoint:
```bash
python train/train_cbae_ddpm_custom.py \
    -d celebahq \
    -e cbae_ddpm \
    -t resumed_training \
    -p clipzs \
    --load-pretrained \
    --pretrained-load-name celebahq_cbae_ddpm_full_4775imgs_8concepts_cbae.pt
```

#### 3. Train with Pseudo-Labels (No Annotations):
Edit config to set `use_real_labels: false`, then:
```bash
python train/train_cbae_ddpm.py \
    -d celebahq \
    -e cbae_ddpm \
    -t pseudolabel_experiment \
    -p clipzs
```

### Training on Different Datasets:

#### CelebA-HQ 256×256 (8 concepts):
```bash
python train/train_cbae_ddpm_custom.py -d celebahq -e cbae_ddpm -t celebahq_run -p clipzs
```

#### CelebA 64×64 (8 concepts):
```bash
python train/train_cbae_ddpm_custom.py -d celeba64 -e cbae_ddpm -t celeba64_run -p clipzs
```

#### CUB-200 64×64 (10 concepts):
```bash
python train/train_cbae_ddpm_custom.py -d cub64 -e cbae_ddpm -t cub_run -p clipzs
```

---

## Training Pipeline

### Training Process Overview:

The training script performs the following steps per epoch:

#### **Phase 1: CB-AE Reconstruction + Concept Alignment + Noise Prediction**

For each batch:

1. **Load clean images** `x_0` and labels from dataset
2. **Sample random timesteps** `t ~ Uniform(0, max_timestep)`
3. **Add noise** using DDPM forward process: `x_t = √(α_t) * x_0 + √(1-α_t) * ε`
4. **Extract noisy latent** `w_t` from UNet encoder (g1)
5. **Encode to concepts**: `c = Encoder(w_t)`
6. **Decode back**: `w_t' = Decoder(c)`
7. **Predict noise**: `ε_pred = UNet_decoder(w_t', t, residuals)`

**Three losses are computed:**
```python
# Loss 1: Latent reconstruction (MSE)
L_recon = ||w_t - w_t'||²

# Loss 2: Concept alignment (CrossEntropy)
L_concept = CE(c, labels)

# Loss 3: Noise prediction (MSE)
L_noise = ||ε - ε_pred||²

# Total loss
L_total = L_recon + L_concept + L_noise
```

**Optimizer:** Adam with learning rate `recon_lr = 0.0002`

#### **Phase 2: Intervention Training**

For each batch:

1. **Sample new timesteps** and add noise to get `x_t'`
2. **Extract latent** `w_t'` from UNet encoder
3. **Encode to concepts**: `c = Encoder(w_t')`
4. **Intervene on random concept**: Swap concept value (e.g., make "Smiling" = 1)
5. **Decode intervened concepts**: `w_intervened = Decoder(c_intervened)`
6. **Predict noise**: `ε_pred = UNet_decoder(w_intervened, t, residuals)`

**Intervention loss:**
```python
# Ensure intervened concepts align with target
L_intervention = CE(c_intervened, target_labels) + ||ε - ε_pred||²
```

**Optimizer:** Separate Adam with learning rate `conc_lr = 0.0002`

### Why Two Training Phases?

1. **Phase 1** ensures the CB-AE learns to:
   - Accurately encode latents to concepts
   - Reconstruct latents without losing information
   - Maintain DDPM's noise prediction quality

2. **Phase 2** ensures the CB-AE decoder can:
   - Handle intervened/modified concept vectors
   - Produce valid latents even with concept edits
   - Enable controllable generation at inference

### Logging and Checkpoints:

**TensorBoard Logs:**
```bash
tensorboard --logdir runs/
```

**Metrics logged:**
- `train/total_loss` - Combined loss
- `train/recon_loss` - Latent reconstruction loss
- `train/concept_loss` - Concept alignment loss
- `train/noise_loss` - Noise prediction loss
- `train/interv_loss` - Intervention loss
- `train/lr` - Learning rate

**Checkpoints saved to:**
```
models/checkpoints/<dataset>_<expt_name>_<tensorboard_name>_cbae.pt
```

Example: `models/checkpoints/celebahq_cbae_ddpm_full_4775imgs_8concepts_cbae.pt`

---

## Monitoring Training

### Using TensorBoard:
```bash
tensorboard --logdir runs/ --port 6006
```
Then open: `http://localhost:6006`

### Key Metrics to Watch:

1. **Concept Loss** should decrease steadily
   - If it plateaus early, consider:
     - Increasing `emb_size`
     - Reducing `batch_size` (more gradient updates)
     - Checking label quality

2. **Reconstruction Loss** should be low (< 0.1)
   - High values indicate the decoder can't reconstruct latents well

3. **Noise Loss** should match original DDPM performance
   - If it's much higher, CB-AE may be destroying latent information

4. **Intervention Loss** should decrease
   - Indicates the model is learning to handle concept interventions

---

## Troubleshooting

### Common Issues:

#### 1. Out of Memory (OOM):
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `batch_size` in config (try 2 or 1)
- Reduce `emb_size` (try 64 instead of 128)
- Use gradient accumulation (modify training script)
- Use smaller image size (64 instead of 256)

#### 2. FileNotFoundError - Annotation file:
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/train_balanced.txt'
```

**Solutions:**
- Check `train_anno` path in config
- Ensure annotation file exists and is readable
- Set `use_real_labels: false` to use pseudo-labels instead

#### 3. Concept Loss is NaN:
```
concept_loss: nan
```

**Solutions:**
- Check label format in annotation file (must be 0 or 1 for binary)
- Ensure number of concepts matches between:
  - Annotation file (number of columns)
  - Config `concept_names` (number of names)
  - Config `concept_bins` (number of bins)

#### 4. Model not improving:
```
Losses not decreasing after many epochs
```

**Solutions:**
- Increase learning rate: `recon_lr: 0.0005`, `conc_lr: 0.0005`
- Train longer: `epochs: 100`
- Check if labels are correct (print first batch labels)
- Ensure `use_real_labels: true` if you have annotations

#### 5. Diffusers version mismatch:
```
AttributeError: 'UNet2DModel' object has no attribute 'forward_part1'
```

**Solution:**
- Our code uses modified UNet2D from `models/cbae_unet2d.py`
- The modification splits UNet into encoder/decoder parts
- This is expected and correct

---

## Advanced Configuration

### Training with Different Number of Concepts:

To train with 4 concepts instead of 8:

```yaml
model:
  concepts:
    concept_bins: [2, 2, 2, 2]
    concept_names: ["attractive", "smiling", "male", "young"]
    emb_size: 64  # Smaller for fewer concepts
    concept_output: [2, 2, 2, 2]
    types: ["bin", "bin", "bin", "bin"]
```

Update annotation file to have 4 concept columns.

### Training at Different Timesteps:

```yaml
model:
  max_timestep: 600  # Apply CB-AE for t ≤ 600 (more noisy)
```

**Trade-offs:**
- Lower timesteps (200-400): Less noisy, easier to learn, better accuracy
- Higher timesteps (600-800): More noisy, harder to learn, broader coverage

### Using Different DDPM Models:

```yaml
model:
  pretrained: google/ddpm-celeba-hq-256  # CelebA-HQ
  # OR
  pretrained: google/ddpm-bedroom-256     # LSUN Bedroom
  # OR
  pretrained: google/ddpm-cat-256         # AFHQ Cat
```

---

## Model Checkpoints

### Checkpoint Contents:

The saved `.pt` file contains the CB-AE state dict:
```python
{
  'core.encoder.0.weight': ...,
  'core.encoder.0.bias': ...,
  'core.decoder.0.weight': ...,
  'core.decoder.0.bias': ...,
  ...
}
```

### Loading a Checkpoint:

```python
from models.cbae_ddpm import cbAE_DDPM_Trainable
import yaml
import torch

# Load config
with open('config/cbae_ddpm/celebahq_full.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build model
model = cbAE_DDPM_Trainable(config)

# Load checkpoint
checkpoint_path = 'models/checkpoints/celebahq_cbae_ddpm_full_4775imgs_8concepts_cbae.pt'
model.cbae.load_state_dict(torch.load(checkpoint_path))
model.eval()
```

---

## Next Steps

After training completes:

1. **Evaluate concept accuracy**: Check if CB-AE can predict concepts correctly
2. **Test interventions**: Generate images with concept interventions
3. **Measure steerability**: Quantify how well interventions work
4. **Fine-tune parameters**: Adjust `max_timestep`, `emb_size` based on results

