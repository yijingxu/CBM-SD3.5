# Training Data Design

## Overview

This document describes the training data structure for Concept Bottleneck Autoencoder (CB_AE) models trained on Stable Diffusion 3.5 Medium latent embeddings.

## Data Generation

- **Model**: `stabilityai/stable-diffusion-3.5-medium`
- **Samples per prompt**: 200
- **Total samples**: 800 (4 prompts × 100 samples)
- **Timestep range**: Randomly selected from 4-14 for each sample
- **Transformer block**: Block 12 (out of 24 total blocks)
- **Guidance scale**: Dynamic (4.5 → 10.0)
- **Save dynamic text embedding**: `True` (saves text embedding per sample) or `False` (saves once per prompt)

## File Structure

The file structure depends on the `save_dynamic_text_embedding` hyperparameter:

### When `save_dynamic_text_embedding = False` (Static Mode)

```
YijingCode/TrainingData/
├── metadata.csv                 # The "Map": Links Prompt_ID <-> Sample_ID <-> Timestep
├── text_embeddings/             # Static Store (One per prompt)
│   ├── prompt_001.pt            # Shape: [1, 333, 1536] (BF16)
│   ├── prompt_002.pt
│   ├── prompt_003.pt
│   └── prompt_004.pt
└── image_trajectories/          # Dynamic Store (100 per prompt)
    ├── prompt_001/
    │   ├── sample_000_t07.pt    # Image latent at step 7
    │   ├── sample_000_noise.pt  # Noise prediction at step 7
    │   ├── sample_001_t04.pt    # Image latent at step 4
    │   ├── sample_001_noise.pt  # Noise prediction at step 4
    │   └── ... (100 samples total)
    ├── prompt_002/
    │   └── ... (100 samples)
    ├── prompt_003/
    │   └── ... (100 samples)
    └── prompt_004/
        └── ... (100 samples)
```

### When `save_dynamic_text_embedding = True` (Dynamic Mode)

```
YijingCode/TrainingData/
├── metadata.csv                 # The "Map": Links Prompt_ID <-> Sample_ID <-> Timestep
└── image_trajectories/          # Dynamic Store (100 per prompt)
    ├── prompt_001/
    │   ├── sample_000_t07.pt           # Image latent at step 7
    │   ├── sample_000_t07_text.pt      # Text embedding at step 7
    │   ├── sample_000_noise.pt         # Noise prediction at step 7
    │   ├── sample_001_t04.pt           # Image latent at step 4
    │   ├── sample_001_t04_text.pt      # Text embedding at step 4
    │   ├── sample_001_noise.pt         # Noise prediction at step 4
    │   └── ... (100 samples total)
    ├── prompt_002/
    │   └── ... (100 samples)
    ├── prompt_003/
    │   └── ... (100 samples)
    └── prompt_004/
        └── ... (100 samples)
```

## Data Components

### 1. Metadata CSV (`metadata.csv`)

The metadata file serves as the central index linking all data components. Each row represents one training sample.

**Columns:**
- `prompt_id`: Integer (1-4) - Identifies which prompt was used
- `sample_id`: Integer (0-99) - Identifies which sample of the prompt
- `timestep`: Integer (4-10) - The diffusion timestep at which data was captured
- `concept_a`: String - Ground truth label for Concept A ("Ice Cream" or "Spaghetti")
- `concept_b`: String - Ground truth label for Concept B ("Cone" or "Plate")
- `prompt_text`: String - Full text of the prompt
- `text_embedding_path`: String - Relative path to text embedding file
- `image_latent_path`: String - Relative path to image latent embedding file
- `noise_pred_path`: String - Relative path to noise prediction file

**Example rows:**

When `save_dynamic_text_embedding = False`:
```csv
prompt_id,sample_id,timestep,concept_a,concept_b,prompt_text,text_embedding_path,image_latent_path,noise_pred_path
1,0,7,Ice Cream,Cone,"A scoop of ice cream served in a crunchy cone.",text_embeddings/prompt_001.pt,image_trajectories/prompt_001/sample_000_t07.pt,image_trajectories/prompt_001/sample_000_noise.pt
```

When `save_dynamic_text_embedding = True`:
```csv
prompt_id,sample_id,timestep,concept_a,concept_b,prompt_text,text_embedding_path,image_latent_path,noise_pred_path
1,0,7,Ice Cream,Cone,"A scoop of ice cream served in a crunchy cone.",image_trajectories/prompt_001/sample_000_t07_text.pt,image_trajectories/prompt_001/sample_000_t07.pt,image_trajectories/prompt_001/sample_000_noise.pt
```

### 2. Text Embeddings

**Purpose**: Text embeddings processed by transformer block 12.

**Two modes based on `save_dynamic_text_embedding` hyperparameter:**

#### Mode 1: Static (`save_dynamic_text_embedding = False`)

**Location**: `text_embeddings/` directory

**Characteristics:**
- **One file per prompt** (not duplicated across samples)
- **Format**: PyTorch tensor (`.pt`)
- **Shape**: `[1, 333, 1536]`
  - `1`: Batch dimension
  - `333`: Sequence length (text tokens)
  - `1536`: Hidden dimension (`caption_projection_dim`)
- **Dtype**: `torch.float32` (converted from bfloat16 for compatibility)
- **Content**: Conditional branch only (text-guided embeddings)
- **Usage**: Input to Text_CB_AE model

**File naming**: `prompt_{prompt_id:03d}.pt`

**Rationale**: Text embeddings are identical for all samples of the same prompt, so storing once per prompt saves storage space.

#### Mode 2: Dynamic (`save_dynamic_text_embedding = True`)

**Location**: `image_trajectories/prompt_{id}/` directory (same as image latents)

**Characteristics:**
- **100 files per prompt** (one per sample, saved at the captured timestep)
- **Format**: PyTorch tensor (`.pt`)
- **Shape**: `[1, 333, 1536]`
  - `1`: Batch dimension
  - `333`: Sequence length (text tokens)
  - `1536`: Hidden dimension (`caption_projection_dim`)
- **Dtype**: `torch.float32` (converted from bfloat16 for compatibility)
- **Content**: Conditional branch only (text-guided embeddings)
- **Usage**: Input to Text_CB_AE model, allows tracking text embeddings at different timesteps

**File naming**: `sample_{sample_id:03d}_t{timestep:02d}_text.pt`

**Rationale**: Enables analysis of text embeddings at different timesteps, useful for understanding how text representations evolve during the diffusion process.

### 3. Image Latent Embeddings (`image_trajectories/prompt_{id}/sample_{id}_t{timestep}.pt`)

**Purpose**: Spatial latent embeddings from transformer block 12 at a specific timestep.

**Characteristics:**
- **100 files per prompt** (one per sample)
- **Format**: PyTorch tensor (`.pt`)
- **Shape**: `[1, 2304, 1536]`
  - `1`: Batch dimension
  - `2304`: Sequence length (spatial patches: 96×96/4 with patch_size=2)
  - `1536`: Hidden dimension (`caption_projection_dim`)
- **Dtype**: `torch.float32` (converted from bfloat16)
- **Content**: Conditional branch only (text-guided image latents)
- **Usage**: Input to Image_CB_AE model

**File naming**: `sample_{sample_id:03d}_t{timestep:02d}.pt`

**Note**: The sequence length 2304 comes from:
- Spatial latent size: 96×96 = 9,216 positions
- Patch size: 2×2
- Patches: (96/2) × (96/2) = 48×48 = 2,304 patches

### 4. Noise Predictions (`image_trajectories/prompt_{id}/sample_{id}_noise.pt`)

**Purpose**: Noise prediction from the transformer at the captured timestep.

**Characteristics:**
- **100 files per prompt** (one per sample)
- **Format**: PyTorch tensor (`.pt`)
- **Shape**: `[1, 16, 96, 96]`
  - `1`: Batch dimension
  - `16`: Latent channels
  - `96×96`: Spatial dimensions
- **Dtype**: `torch.float32`
- **Content**: Conditional branch only (text-guided noise prediction)
- **Usage**: Reference/analysis for the denoising process

**File naming**: `sample_{sample_id:03d}_noise.pt`

## Concept Labels

The dataset includes two binary concepts:

### Concept A: Food Type
- **"Ice Cream"**: Prompts 1, 2
- **"Spaghetti"**: Prompts 3, 4

### Concept B: Container Type
- **"Cone"**: Prompts 1, 4
- **"Plate"**: Prompts 2, 3

**Prompt Mapping:**
1. Prompt 1: Ice Cream + Cone
2. Prompt 2: Ice Cream + Plate
3. Prompt 3: Spaghetti + Plate
4. Prompt 4: Spaghetti + Cone

## Design Rationale

### Why This Structure?

1. **Metadata + Flat Tensor Structure**: 
   - Centralized metadata makes it easy to query and filter data
   - Flat tensor files are efficient for loading during training

2. **Text Embedding Storage Modes**:
   - **Static mode** (`save_dynamic_text_embedding = False`): Text embeddings are identical for all samples of the same prompt, so storing once per prompt saves ~400MB of storage (100 samples × 4MB per embedding)
   - **Dynamic mode** (`save_dynamic_text_embedding = True`): Text embeddings are saved per sample, enabling analysis of text representations at different timesteps, useful for understanding timestep-dependent text embedding variations
   - Metadata links samples to their text embedding regardless of mode

3. **Organized by Prompt**:
   - Easy to find all samples for a specific prompt
   - Natural grouping for analysis and debugging

4. **Timestep in Filename**:
   - Quick identification of when data was captured
   - Useful for analyzing timestep-dependent patterns

5. **PyTorch Format (`.pt`)**:
   - Native PyTorch format preserves bfloat16 correctly
   - Faster loading than NumPy format
   - No dtype conversion issues

### Why Conditional Branch Only?

- **Concept Information**: Conditional branch contains text-guided representations with concept information
- **Training Focus**: CB_AE models learn concept-specific features from conditional embeddings
- **Storage Efficiency**: Unconditional branch is only needed for CFG during generation, not for training
- **Clarity**: Single branch simplifies training data structure

## Data Loading Example

```python
import torch
import pandas as pd

# Load metadata
metadata = pd.read_csv("YijingCode/TrainingData/metadata.csv")

# Load a specific sample
row = metadata.iloc[0]
text_emb = torch.load(f"YijingCode/TrainingData/{row['text_embedding_path']}")
image_latent = torch.load(f"YijingCode/TrainingData/{row['image_latent_path']}")
noise_pred = torch.load(f"YijingCode/TrainingData/{row['noise_pred_path']}")

# Access ground truth concepts
concept_a = row['concept_a']  # "Ice Cream" or "Spaghetti"
concept_b = row['concept_b']  # "Cone" or "Plate"
```

## Statistics

### When `save_dynamic_text_embedding = False` (Static Mode)

- **Total files**: 804
  - 4 text embedding files
  - 400 image latent files (100 per prompt)
  - 400 noise prediction files (100 per prompt)
- **Total samples**: 400
- **Timestep distribution**: Uniform random from 4-10
- **Storage per sample**: ~4MB (text, shared) + ~14MB (image) + ~0.6MB (noise) ≈ 19MB
- **Total storage**: ~7.6GB (4 text + 400 image + 400 noise files)

### When `save_dynamic_text_embedding = True` (Dynamic Mode)

- **Total files**: 1200
  - 400 text embedding files (100 per prompt)
  - 400 image latent files (100 per prompt)
  - 400 noise prediction files (100 per prompt)
- **Total samples**: 400
- **Timestep distribution**: Uniform random from 4-10
- **Storage per sample**: ~4MB (text) + ~14MB (image) + ~0.6MB (noise) ≈ 19MB
- **Total storage**: ~7.6GB (400 text + 400 image + 400 noise files)

## Notes

- All tensors are saved in CPU memory (detached from computation graph)
- Text embeddings are converted to float32 for compatibility
- Image latents and noise predictions are also float32
- File paths in metadata are relative to `YijingCode/TrainingData/` directory
- Sample IDs are zero-indexed (0-99)
- Prompt IDs are one-indexed (1-4)
- The `save_dynamic_text_embedding` hyperparameter controls whether text embeddings are saved once per prompt (static) or per sample (dynamic)
- In dynamic mode, text embeddings are saved alongside image latents in the same directory with the `_text.pt` suffix

