import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ==========================================
# 1. Utilities & Gradient Reversal
# ==========================================

def _weights_init(m):
    """Initialize weights for linear and convolutional layers."""
    classname = m.__class__.__name__
    if 'Linear' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradients: multiply by -alpha
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# ==========================================
# 2. Stream Implementations (Text & Image)
# ==========================================

class Text_CB_AE(nn.Module):
    """
    Specialized for Text Stream (C_ctxt).
    Uses Mean Pooling to compress the sequence into a semantic vector
    before bottlenecking, enforcing global semantic constraints.
    
    Modes:
    - use_adversarial=True  -> Soft Bottleneck (Concepts + Residual Stream).
                               Includes Gradient Reversal to purge concepts from residual.
    - use_adversarial=False -> Hard Bottleneck (Concepts Only).
                               Residual stream is removed entirely.
    
    Default dimensions match SD3.5 Medium: seq_len=333, embed_dim=1536
    """
    def __init__(self, seq_len=333, embed_dim=1536, concept_dim=2, hidden_dim=4096, use_adversarial=False):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.use_adversarial = use_adversarial
        
        # LOGIC: If no adversarial training, we force a Hard Bottleneck (0 residual).
        # If adversarial is ON, we allow a small residual stream (16 dim) to aid reconstruction,
        # but we will police it with the discriminator.
        if self.use_adversarial:
            self.unsupervised_dim = 16 
        else:
            self.unsupervised_dim = 0
            
        self.total_bottleneck = concept_dim + self.unsupervised_dim
        
        # --- Encoder ---
        # Input is the pooled embedding (embed_dim), not the full sequence
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # Added depth for complex SD3 signals
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, self.total_bottleneck) 
        )
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(self.total_bottleneck, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )

        # --- Adversarial Branch (Only for Soft Bottleneck) ---
        if self.use_adversarial:
            self.grl = GradientReversal(alpha=1.0)
            # Predicts concepts from the residual to maximize error
            self.residual_classifier = nn.Sequential(
                nn.Linear(self.unsupervised_dim, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, concept_dim) 
            )
            
        self.apply(_weights_init)

    def forward(self, x):
        """
        Forward pass for text embeddings.
        
        Args:
            x: Text embeddings, shape (Batch, seq_len, embed_dim)
               e.g., (Batch, 333, 1536) for SD3.5 Medium
        
        Returns:
            recon: Reconstructed embeddings, shape (Batch, seq_len, embed_dim)
            c: Concept logits, shape (Batch, concept_dim)
            adv_logits: (Optional) Adversarial logits if use_adversarial=True
        """
        # x shape: (Batch, seq_len, embed_dim) e.g., (Batch, 333, 1536) for SD3.5 Medium
        
        # 1. POOLING: Compress sequence to get global semantics
        x_pooled = x.mean(dim=1) # (Batch, embed_dim)
        
        # 2. Encode -> Latent z
        z = self.encoder(x_pooled)
        
        # 3. Split Concepts (c) and Residual (u)
        if self.unsupervised_dim > 0:
            c = z[:, :-self.unsupervised_dim] 
            u = z[:, -self.unsupervised_dim:]
        else:
            # Hard Bottleneck: The entire z is the concept vector
            c = z
            u = None 
        
        # 4. Adversarial Pass (if enabled)
        adv_logits = None
        if self.use_adversarial and u is not None:
            # GRL reverses gradient here to confuse the classifier
            u_reversed = self.grl(u)
            adv_logits = self.residual_classifier(u_reversed)
        
        # 5. Decode
        recon_pooled = self.decoder(z) # (Batch, embed_dim)
        
        # 6. UNPOOLING / BROADCASTING
        # We must return (Batch, seq_len, embed_dim) for the MM-DiT.
        # Since we mean-pooled, we expand the reconstructed vector back to the sequence length.
        # 
        # DESIGN CHOICE: This is a STRONG CONSTRAINT that forces every text token to 
        # reconstruct to the exact same pooled concept vector. This ensures the global 
        # concept (e.g., "Concept: Spooky") permeates the entire prompt embedding so 
        # MM-DiT attention layers cannot ignore it.
        # 
        # If this proves too restrictive (destroys detailed syntax of the prompt), 
        # consider relaxing the bottleneck: instead of a single vector, map to a sequence 
        # of k concept tokens (e.g., one per sentence/phrase). For now, the "Global 
        # Semantic" approach is the standard starting point for steerability.
        recon = recon_pooled.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        if self.use_adversarial:
            return recon, c, adv_logits
        return recon, c

class Image_CB_AE(nn.Module):
    """
    Spatial Concept Bottleneck Autoencoder for SD3.5 Image Stream.
    
    Instead of a massive MLP, this uses Conv2d layers to process the 
    spatial grid (48x48 patches), dramatically reducing parameter count 
    while preserving spatial context.
    
    Default dimensions match SD3.5 Medium: num_patches=2304, patch_dim=1536
    Input: (Batch, num_patches, patch_dim) e.g., (B, 2304, 1536)
    Output: (Batch, num_patches, patch_dim) - reconstructed image patches
    """
    def __init__(self, num_patches=2304, patch_dim=1536, concept_dim=2, use_adversarial=False):
        super().__init__()
        self.use_adversarial = use_adversarial
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        
        # Calculate spatial side length: sqrt(2304) = 48
        self.spatial_side = int(num_patches**0.5) 
        
        # Bottleneck logic
        if self.use_adversarial:
            self.unsupervised_dim = 16 
        else:
            self.unsupervised_dim = 0
        self.total_bottleneck = concept_dim + self.unsupervised_dim

        # --- ENCODER (Conv2d) ---
        # Input: (B, 1536, 48, 48)
        self.encoder = nn.Sequential(
            # Downsample 48x48 -> 24x24
            nn.Conv2d(patch_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            # Downsample 24x24 -> 12x12
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            # Global Average Pooling (12x12 -> 1x1)
            # This collapses spatial dims to get a single concept vector
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Projection to Concept Space
            nn.Linear(256, self.total_bottleneck)
        )

        # --- DECODER (Transposed Conv) ---
        # Map Concept -> Spatial Feature Map
        self.decoder_input = nn.Sequential(
            nn.Linear(self.total_bottleneck, 256 * 12 * 12),
            nn.LeakyReLU(0.1)
        )
        
        self.decoder_conv = nn.Sequential(
            # Input: (B, 256, 12, 12) -> Upsample to 24x24
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            # Upsample 24x24 -> 48x48
            nn.ConvTranspose2d(512, patch_dim, kernel_size=4, stride=2, padding=1),
            # No activation at output (linear reconstruction)
        )

        # --- ADVERSARIAL BRANCH ---
        if self.use_adversarial:
            self.grl = GradientReversal(alpha=1.0)
            self.residual_classifier = nn.Sequential(
                nn.Linear(self.unsupervised_dim, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, concept_dim) 
            )
        
        # Initialize weights
        self.apply(_weights_init)

    def forward(self, x):
        """
        Forward pass for image patches.
        
        Args:
            x: Image patches, shape (Batch, num_patches, patch_dim)
               e.g., (B, 2304, 1536) for SD3.5 Medium
        
        Returns:
            recon: Reconstructed patches, shape (Batch, num_patches, patch_dim)
            c: Concept logits, shape (Batch, concept_dim)
            adv_logits: (Optional) Adversarial logits if use_adversarial=True
        """
        # x input shape: (Batch, 2304, 1536)
        batch_size = x.shape[0]
        
        # 1. RESHAPE TO IMAGE (B, Dim, H, W)
        # We perform a transpose to get channels first
        x_spatial = x.transpose(1, 2).view(batch_size, self.patch_dim, self.spatial_side, self.spatial_side)
        
        # 2. ENCODE
        z = self.encoder(x_spatial) # (Batch, total_bottleneck)
        
        # 3. SPLIT
        if self.unsupervised_dim > 0:
            c = z[:, :-self.unsupervised_dim] 
            u = z[:, -self.unsupervised_dim:]
        else:
            c = z
            u = None 
            
        # 4. ADVERSARIAL
        adv_logits = None
        if self.use_adversarial and u is not None:
            u_reversed = self.grl(u)
            adv_logits = self.residual_classifier(u_reversed)
            
        # 5. DECODE
        # Expand vector back to 12x12 feature map
        rec_feat = self.decoder_input(z)
        rec_feat = rec_feat.view(batch_size, 256, 12, 12)
        
        # Upsample back to 48x48
        recon_spatial = self.decoder_conv(rec_feat)
        
        # 6. RESHAPE BACK TO SEQUENCE (B, 2304, 1536)
        # Flatten spatial dims and transpose back
        recon = recon_spatial.flatten(2).transpose(1, 2)
        
        if self.use_adversarial:
            return recon, c, adv_logits
        return recon, c


# ==========================================
# 4. Objective Function (The "DualStreamLoss")
# ==========================================

class DualStreamLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_concept=1.0, lambda_consist=5.0, lambda_adv=0.5):
        super().__init__()
        self.lambda_recon = lambda_recon     # Trade-off for reconstruction quality
        self.lambda_concept = lambda_concept # Trade-off for concept accuracy
        self.lambda_consist = lambda_consist # Trade-off for locking streams together
        self.lambda_adv = lambda_adv         # Trade-off for disentanglement
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss() # Assuming multi-label binary concepts

    def forward(self, 
                img_out, txt_out, 
                target_img, target_txt, 
                pseudo_labels):
        """
        Args:
            img_out: Tuple (recon_img, c_img, adv_img_logits [opt])
            txt_out: Tuple (recon_txt, c_txt, adv_txt_logits [opt])
            target_img: Original Image Latents
            target_txt: Original Text Embeddings
            pseudo_labels: Shared ground truth concepts (Batch, Concept_Dim)
        """
        
        # Unpack outputs
        recon_img, c_img = img_out[0], img_out[1]
        recon_txt, c_txt = txt_out[0], txt_out[1]
        
        # --------------------------------------------
        # 1. RECONSTRUCTION LOSS (L_r)
        # Ensure MM-DiT can still generate good images/understand prompts
        # --------------------------------------------
        loss_r_img = self.mse(recon_img, target_img)
        loss_r_txt = self.mse(recon_txt, target_txt)
        loss_recon = loss_r_img + loss_r_txt
        
        # --------------------------------------------
        # 2. CONCEPT ALIGNMENT LOSS (L_c)
        # Both streams must predict the correct concept from the pseudo-labels
        # 
        # NOTE: Using BCEWithLogitsLoss assumes MULTI-LABEL (independent) concepts.
        # This is appropriate for binary attributes (e.g., CelebA: "Male", "Smiling", 
        # "Glasses" can all be true simultaneously).
        # 
        # If your concepts are MUTUALLY EXCLUSIVE (categorical), switch to CrossEntropyLoss:
        #   ce_loss = nn.CrossEntropyLoss()
        #   loss_c_img = ce_loss(c_img, pseudo_labels.long())  # pseudo_labels should be class indices
        #   loss_c_txt = ce_loss(c_txt, pseudo_labels.long())
        # 
        # The original paper uses a MIX: 2 logits per binary concept + N logits per categorical concept.
        # For steerability (attribute control), multi-label BCE is the standard approach.
        # --------------------------------------------
        loss_c_img = self.bce(c_img, pseudo_labels)
        loss_c_txt = self.bce(c_txt, pseudo_labels)
        loss_concept = loss_c_img + loss_c_txt
        
        # --------------------------------------------
        # 3. CROSS-MODAL CONSISTENCY LOSS (L_consist)
        # CRITICAL: Force Text Concept Vector == Image Concept Vector
        # This assures that "concept #3" in text means the same as in image.
        # --------------------------------------------
        loss_consist = self.mse(c_img, c_txt)
        
        # --------------------------------------------
        # 4. ADVERSARIAL LOSS (L_adv) [Optional]
        # If enabled, train residual classifier to predict concepts.
        # Because of GRL, minimizing this maximizes encoder confusion.
        # --------------------------------------------
        loss_adv = torch.tensor(0.0, device=recon_img.device)
        
        # Check if adversarial logits exist (index 2 of tuple)
        if len(img_out) > 2 and img_out[2] is not None:
            adv_img = img_out[2]
            loss_adv += self.bce(adv_img, pseudo_labels)
            
        if len(txt_out) > 2 and txt_out[2] is not None:
            adv_txt = txt_out[2]
            loss_adv += self.bce(adv_txt, pseudo_labels)

        # --------------------------------------------
        # TOTAL LOSS
        # --------------------------------------------
        total_loss = (self.lambda_recon * loss_recon) + \
                     (self.lambda_concept * loss_concept) + \
                     (self.lambda_consist * loss_consist) + \
                     (self.lambda_adv * loss_adv)
                     
        return {
            "total": total_loss,
            "recon": loss_recon.item(),
            "concept": loss_concept.item(),
            "consistency": loss_consist.item(), # Monitor this closely!
            "adversarial": loss_adv.item()
        }

# ==========================================
# Example Usage Script
# ==========================================
if __name__ == "__main__":
    # Hyperparams - Updated to match SD3.5 Medium dimensions
    BS = 4
    SEQ_LEN = 333      # Text sequence length for SD3.5 Medium
    PATCHES = 2304     # Image patches for SD3.5 Medium (96x96/4 with patch_size=2)
    DIM = 1536         # Hidden dimension (caption_projection_dim) for SD3.5 Medium
    CONCEPTS = 2       # Number of concepts (concept_a: Food, concept_b: Container)
    
    # 1. Init Models (Toggle use_adversarial=True/False here)
    # Note: False = Hard Bottleneck (0 residual)
    use_adv = False 
    
    text_cbm = Text_CB_AE(seq_len=SEQ_LEN, embed_dim=DIM, concept_dim=CONCEPTS, use_adversarial=use_adv)
    image_cbm = Image_CB_AE(num_patches=PATCHES, patch_dim=DIM, concept_dim=CONCEPTS, use_adversarial=use_adv)
    
    loss_fn = DualStreamLoss()
    
    # 2. Dummy Data (simulating a training step)
    # Original latent inputs from Frozen SD3
    real_txt_emb = torch.randn(BS, SEQ_LEN, DIM) 
    real_img_lat = torch.randn(BS, PATCHES, DIM)
    
    # Shared Pseudo-labels (e.g., from CLIP running on the generated image)
    # Multi-hot vector (e.g., [1, 0, 1...] for Male, Not-Smiling, ...)
    pseudo_labels = torch.randint(0, 2, (BS, CONCEPTS)).float()
    
    # 3. Forward Pass
    txt_out = text_cbm(real_txt_emb)
    img_out = image_cbm(real_img_lat)
    
    # 4. Calculate Loss
    # Notice we pass the outputs and the original targets for reconstruction
    losses = loss_fn(img_out, txt_out, real_img_lat, real_txt_emb, pseudo_labels)
    
    # 5. Backward
    losses["total"].backward()
    
    print("Dual-Stream CBM Training Step Complete.")
    print(f"Mode: {'Soft/Adversarial' if use_adv else 'Hard Bottleneck'}")
    print("Losses:", losses)