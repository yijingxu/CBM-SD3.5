# DDPM + Concept Bottleneck Autoencoder (CB-AE)

This repository contains code to train a **Concept Bottleneck
Autoencoder (CB-AE)** inserted at the **UNet mid-block of a pretrained
DDPM model**, following the **Post-hoc Generative Concept Bottleneck
Model (CBGM)** framework.

The implementation adapts the original CB-AE design from the paper’s
released code (`cbae_unet2d.py`) and integrates it with a **pretrained
DDPM UNet** from Hugging Face, while keeping the diffusion backbone
frozen.

------------------------------------------------------------------------

## Overview

-   **Backbone**: pretrained DDPM UNet (`UNet2DModel`)
-   **CB-AE insertion point**: UNet mid-block latent representation
-   **Trainable parameters**: CB-AE only (DDPM UNet is frozen)
-   **Training objective**:
    -   DDPM noise prediction consistency
    -   Supervised concept prediction loss
    -   Latent reconstruction losses

This setup follows the post-hoc philosophy of CBGM: interpretability is
introduced *after* generative pretraining, without modifying the
diffusion backbone.

------------------------------------------------------------------------

## Quick Start

### 1. Environment setup

We recommend using a CUDA-enabled environment.

``` bash
# Clone the repository
git clone https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm.git
cd posthoc-generative-cbm

# Install PyTorch (CUDA 12.1)
pip install -q torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# Install CLIP and Hugging Face dependencies
pip install -q git+https://github.com/openai/CLIP.git
pip install -q transformers huggingface_hub accelerate datasets

# Install project-specific requirements
pip install -q -r requirements.txt

# Add the repository to the Python path
export PYTHONPATH="$PYTHONPATH:/content/posthoc-generative-cbm"
```

### 2. Train CB-AE (2000 steps)

``` bash
python scripts/train_cbae_ddpm.py \
  --dataset_id flwrlabs/celeba \
  --split train \
  --batch_size 4 \
  --total_steps 2000 \
  --log_every 200 \
  --save_every 500 \
  --pretrained_model_id google/ddpm-celebahq-256
```

### 3. Evaluate concept accuracy

``` bash
python scripts/eval_concept_accuracy.py \
  --dataset_id flwrlabs/celeba \
  --split valid \
  --eval_mode random_t \
  --eval_batches 2000
```

------------------------------------------------------------------------

## Dataset

### Training and Evaluation

The current implementation uses CelebA (256×256) images with attribute
annotations.

Each training batch provides:

-   `x0`: clean images of shape `[B, 3, 256, 256]`, normalized to
    `[-1, 1]`
-   `y01`: binary concept labels of shape `[B, C]`

Concept labels are obtained from:

-   CelebA attribute annotations (e.g., `Smiling`, `Young`, `Male`,
    `Eyeglasses`)
-   CLIP-based pseudo-labels (see `clip_pseudolabeler.py` in the
    original repository)

> **Note**\
> **CLIP is not used for concept supervision or concept accuracy
> evaluation.**\
> CLIP is used **only for steerability evaluation** in later stages of
> the CBGM pipeline,\
> where generated images do not have ground-truth concept labels.

Image preprocessing (center crop, resize, normalization) must be
consistent with the preprocessing expected by the pretrained DDPM model.

------------------------------------------------------------------------

## Pretrained Models

We use a pretrained DDPM UNet from Hugging Face:

-   **DDPM (CelebA-HQ, 256×256)**\
    `google/ddpm-celebahq-256`

Although the model identifier references CelebA-HQ, the UNet is treated
as a **generic pretrained 256×256 face diffusion backbone**. CB-AE
training and evaluation are performed using CelebA attributes.

------------------------------------------------------------------------

## Training Details

-   **Frozen backbone**:\
    The pretrained DDPM UNet parameters are **not updated** during
    training.

-   **Optimized module**:\
    Only the **Concept Bottleneck Autoencoder (CB-AE)** parameters are
    trained.

-   **Loss components**:

    -   **L**<sub>r1</sub>: Reconstruction loss on **UNet mid-block
        latent features**
    -   **L**<sub>r2</sub>: Proxy loss preserving **DDPM noise
        prediction behavior**
    -   **L**<sub>c</sub>: Supervised **concept prediction loss**

This corresponds to the CB-AE objective in the original paper,
**excluding intervention losses**.

------------------------------------------------------------------------

## Concept Accuracy Evaluation

Concept accuracy is evaluated on the **CelebA validation split**,
following the evaluation protocol used for **Table 2** in the original
paper.

-   Accuracy is computed using **ground-truth CelebA attribute labels**
-   No hyperparameter tuning is performed based on validation accuracy
-   The validation split is used **only for evaluation**, not for model
    selection

This evaluation serves as a **diagnostic check of concept alignment**
for the trained CB-AE.

------------------------------------------------------------------------

## Reproducibility

The reported results were obtained using:

-   **Training**: `scripts/train_cbae_ddpm.py`
-   **Evaluation**: `scripts/eval_concept_acc.py`

All experiments were run in a Google Colab environment with GPU support.

------------------------------------------------------------------------

## Hugging Face Authentication

Some environments may require logging in to Hugging Face for downloading
or uploading checkpoints.

``` bash
huggingface-cli login
```
