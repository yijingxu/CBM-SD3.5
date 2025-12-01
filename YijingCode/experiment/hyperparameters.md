# Stable Diffusion 3.5 Medium - Hyperparameter Configuration

## Decision Summary

This document records the hyperparameter configuration decisions for experiments using Stable Diffusion 3.5 Medium.

## Hyperparameters

### Inference Steps
- **Value**: `16 steps`
- **Rationale**: Balanced between generation quality and computational efficiency

### Guidance Scale
- **Type**: Dynamic guidance scale
- **Enabled**: `True`
- **Start Scale**: `4.5`
- **End Scale**: `10.0`
- **Rationale**: 
  - Dynamic guidance allows adaptive control throughout the denoising process
  - Starting at 4.5 provides more creative freedom in early steps
  - Ramping up to 10.0 ensures strong prompt adherence in later steps
  - This approach helps avoid artifacts from high static guidance while maintaining prompt fidelity

### Transformer Block Selection
- **Block Index**: `12` (out of 24 total blocks)
- **Rationale**: Mid-layer block captures meaningful representations

### Timestep Selection
- **Target Timestep Index**: `8` (out of 16 total steps)
- **Rationale**: Mid-point of the denoising process for inspection

## Model Configuration

- **Model**: `stabilityai/stable-diffusion-3.5-medium`
- **Image Resolution**: 768x768 (latents: 96x96)
- **Latent Channels**: 16
- **Transformer Blocks**: 24
- **Patch Size**: 2

## Output Streams

From transformer block 12:
- **Text Stream**: `[2, 333, 1536]` - Use for Text_CB_AE
- **Image Stream**: `[2, 2304, 1536]` - Use for Image_CB_AE
  - Sequence length 2304 = 96×96/4 (with patch_size=2)

## Notes

- This configuration is used in `latent_shape_inspection.py`
- Images are saved to the `eg/` directory with descriptive filenames including prompt, steps, and guidance scale info

