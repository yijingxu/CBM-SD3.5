Hi team,

I've just completed a major refactoring of our CBL (Concept Bottleneck Learning) codebase and wanted to share what I've done, plus flag an important issue that needs attention.

## What I've Done

I've modularized and restructured the codebase to make it more maintainable and easier to work with:

1. **Create `CBmodel.py`**: 
   - `Text_CB_AE`: Text stream concept bottleneck autoencoder
   - `Image_CB_AE`: Image stream concept bottleneck autoencoder (now using Conv2d for efficiency)
   - `DualStreamLoss`: Unified loss function for both streams

2. **Added new modules**:
   - `train.py`: Complete training pipeline with dual-stream models, validation, checkpointing, and early stopping
   - `inference.py`: Inference and intervention capabilities for concept manipulation
   - `dataset.py`: `DualStreamDataset` for handling text/image latent pairs
   - `__init__.py`: Package initialization

3. **Updated training data generation** (`latent_embedding.py`):
   - samples per prompt: 100 → 200
   - timestep range: (4, 10) → (4, 16)
   - Added option to save dynamic text embeddings per sample

4. **Training infrastructure**:
   - Both hard bottleneck and adversarial (soft bottleneck) training modes
   - Comprehensive logging and checkpointing
   - Training configs saved for reproducibility

All changes have been committed and pushed to the main branch.

## Issue: Reconstruction Loss Too High

I've been running training experiments and noticed that the **reconstruction loss is extremely high** (~1170-1186 in validation). While the concept alignment (0.067) and consistency (0.10) losses look good, the reconstruction component is dominating the total loss and likely preventing proper learning.

**Current reconstruction loss implementation** (in `CBmodel.py`, lines 341-343):
- Uses simple MSE loss: `loss_r_img = self.mse(recon_img, target_img)` and `loss_r_txt = self.mse(recon_txt, target_txt)`
- No normalization or scaling applied
- The bottleneck might be too restrictive (especially for the hard bottleneck mode with 0 residual dimensions)

**Potential issues to investigate**:
1. The MSE loss might need normalization (e.g., per-patch/token normalization, or relative to input magnitude)
2. The bottleneck dimensions might be too small for the complexity of SD3.5 embeddings
3. The decoder architecture might need more capacity or different activation functions
4. The reconstruction loss might benefit from a different loss function (e.g., L1, Huber, or perceptual loss)

Could someone take a look at the `CBmodel.py` file and work on reducing the reconstruction loss? The relevant sections are:
- `Text_CB_AE.forward()` (lines 112-169) - text reconstruction
- `Image_CB_AE.forward()` (lines 251-302) - image reconstruction  
- `DualStreamLoss.forward()` (lines 338-343) - reconstruction loss computation

Thanks!

