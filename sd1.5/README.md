# CB-AE for Stable Diffusion 1.5 on CelebA

This repository implements the Concept Bottleneck Autoencoder (CB-AE) from the paper "Interpretable Generative Models through Post-hoc Concept Bottlenecks" for Stable Diffusion 1.5.

## Overview

The CB-AE is inserted at the bottleneck of the SD1.5 UNet to enable:
- **Interpretability**: Predict human-understandable concepts (CelebA attributes) from generated images
- **Steerability**: Modify specific concepts during image generation through interventions

### Architecture

```
Noise z -> VAE Encoder -> Noisy Latent -> UNet Encoder (g1) -> Bottleneck Features
                                                                      |
                                                                    CB-AE
                                                                      |
Bottleneck Features (reconstructed) -> UNet Decoder (g2) -> Noise Prediction
```

Only the CB-AE is trained; the SD1.5 VAE and UNet are frozen.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### CelebA

1. Download CelebA from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Extract to get:
   ```
   celeba/
   ├── img_align_celeba/
   │   ├── 000001.jpg
   │   ├── 000002.jpg
   │   └── ...
   ├── list_attr_celeba.txt
   └── list_eval_partition.txt
   ```

### CelebA-HQ

1. Download CelebA-HQ
2. Ensure the structure is:
   ```
   celebahq/
   ├── images/
   │   ├── 0.jpg
   │   ├── 1.jpg
   │   └── ...
   └── CelebAMask-HQ-attribute-anno.txt (optional)
   ```

## Training

### Basic Usage

Train on CelebA with default 8 balanced attributes:

```bash
python train_cbae_sd15.py \
    --dataset celeba \
    --data-root /path/to/celeba \
    --batch-size 16 \
    --epochs 50 \
    --wandb-project my-cbae-project
```

### Custom Attributes

Specify which attributes to predict:

```bash
python train_cbae_sd15.py \
    --dataset celeba \
    --data-root /path/to/celeba \
    --attributes Smiling Male Eyeglasses Young Bald \
    --batch-size 16 \
    --epochs 50
```

### Available CelebA Attributes

All 40 CelebA attributes are supported:

- **Facial Features**: 5_o_Clock_Shadow, Arched_Eyebrows, Bags_Under_Eyes, Big_Lips, Big_Nose, Bushy_Eyebrows, High_Cheekbones, Narrow_Eyes, Oval_Face, Pointy_Nose, Rosy_Cheeks
- **Hair**: Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Receding_Hairline, Straight_Hair, Wavy_Hair
- **Accessories**: Eyeglasses, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie
- **Makeup**: Heavy_Makeup
- **Expression**: Mouth_Slightly_Open, Smiling
- **Gender/Age**: Male, Young
- **Facial Hair**: Goatee, Mustache, No_Beard, Sideburns
- **Other**: Attractive, Blurry, Chubby, Double_Chin, Pale_Skin

### Common Balanced Attributes (8)

These are recommended for balanced training:
- Smiling, Male, Heavy_Makeup, Mouth_Slightly_Open
- Attractive, Wearing_Lipstick, High_Cheekbones, Wavy_Hair

### Using CLIP Pseudo-Labels

To train without ground truth labels (using CLIP zero-shot predictions):

```bash
python train_cbae_sd15.py \
    --dataset celeba \
    --data-root /path/to/celeba \
    --pseudo-label clipzs \
    --batch-size 16 \
    --epochs 50
```

### Full Options

```bash
python train_cbae_sd15.py --help
```

Key arguments:
- `--dataset`: Dataset name (`celeba` or `celebahq`)
- `--data-root`: Path to dataset
- `--attributes`: List of attributes to predict
- `--img-size`: Image size (256 or 512)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate for reconstruction/concept losses
- `--max-timestep`: Maximum diffusion timestep (default: 400)
- `--pseudo-label`: Label source (`real` or `clipzs`)
- `--wandb-project`: Weights & Biases project name

## Monitoring Training

Training logs are sent to Weights & Biases. Key metrics:

### Training Losses
- `train/recon_loss`: Bottleneck reconstruction loss (L_r1)
- `train/concept_loss`: Concept alignment loss (L_c)
- `train/noise_loss`: Noise prediction loss
- `train/interv_loss`: Intervention consistency loss (L_i2)

### Evaluation Metrics
- `eval/overall_accuracy`: Overall concept prediction accuracy
- `eval/accuracy_{attribute}`: Per-attribute accuracy

## Checkpoints

Checkpoints are saved to `--checkpoint-dir` (default: `checkpoints/`):
- `best_cbae.pt`: Best model by validation accuracy
- `cbae_epoch_{N}.pt`: Periodic checkpoints
- `cbae_final.pt`: Final model

### Loading a Checkpoint

```python
from models.cbae_sd15 import SD15WithCBAE

# Create model
model = SD15WithCBAE(
    n_concepts=8,
    img_size=256,
)

# Load checkpoint
checkpoint = torch.load("checkpoints/best_cbae.pt")
model.cbae.load_state_dict(checkpoint["cbae_state_dict"])
```

## Concept Intervention

After training, you can intervene on concepts to steer image generation:

```python
import torch
from models.cbae_sd15 import SD15WithCBAE, get_concept_index

# Load trained model
model = SD15WithCBAE(n_concepts=8, img_size=256)
checkpoint = torch.load("checkpoints/best_cbae.pt")
model.cbae.load_state_dict(checkpoint["cbae_state_dict"])
model.to("cuda")
model.eval()

# Generate an image and get concepts
latents = torch.randn(1, 4, 32, 32, device="cuda")
timesteps = torch.zeros(1, device="cuda", dtype=torch.long)

bottleneck, cache = model.get_unet_bottleneck(latents, timesteps)
concepts = model.cbae.encode(bottleneck)

# Intervene on "Smiling" (concept 0) - set to positive
start, end = get_concept_index(model, 0)
concepts_intervened = concepts.clone()
concepts_intervened[:, start:end] = torch.tensor([[0.0, 1.0]], device="cuda")

# Decode intervened concepts
bottleneck_intervened = model.cbae.decode(concepts_intervened)
```

## Training Details

### Loss Functions

1. **Reconstruction Loss (L_r1)**: MSE between original and reconstructed bottleneck features
2. **Concept Alignment Loss (L_c)**: Cross-entropy between predicted concepts and ground truth labels
3. **Noise Prediction Loss**: MSE between predicted and actual noise (maintains generation quality)
4. **Intervention Loss (L_i2)**: Ensures intervened concepts are correctly encoded after decoding

### Training Procedure

1. Load clean image from CelebA with ground truth attributes
2. Encode image to VAE latent space
3. Sample random timestep t ∈ [0, max_timestep]
4. Add noise to get noisy latent
5. Extract bottleneck features from frozen UNet encoder
6. Apply CB-AE: encode to concepts, decode to reconstructed features
7. Run frozen UNet decoder for noise prediction
8. Compute and backpropagate losses (only CB-AE parameters are updated)

### Memory Requirements

- 256x256 images, batch size 16: ~16GB VRAM
- 512x512 images, batch size 8: ~24GB VRAM

Reduce batch size if you encounter OOM errors.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{kulkarni2025interpretable,
  title={Interpretable Generative Models through Post-hoc Concept Bottlenecks},
  author={Kulkarni, Akshay and Yan, Ge and Sun, Chung-En and Oikarinen, Tuomas and Weng, Tsui-Wei},
  journal={arXiv preprint arXiv:2503.19377},
  year={2025}
}
```

## License

This project is for research purposes. Please check the licenses of:
- Stable Diffusion 1.5 (CreativeML Open RAIL-M)
- CelebA dataset
- CLIP (MIT License)
