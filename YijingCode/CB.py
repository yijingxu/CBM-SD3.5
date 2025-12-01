import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ==========================================
# 1. Utilities & Gradient Reversal
# ==========================================

def _weights_init(m):
    """Initialize weights for linear layers."""
    classname = m.__class__.__name__
    if 'Linear' in classname:
        nn.init.xavier_uniform_(m.weight.data)
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
# 2. Base Concept Bottleneck Autoencoder
# ==========================================

class CB_AE_Base(nn.Module):
    """
    Base Concept Bottleneck Autoencoder.
    
    Modes:
    - use_adversarial=True  -> Soft Bottleneck (Concepts + Residual Stream).
                               Includes Gradient Reversal to purge concepts from residual.
    - use_adversarial=False -> Hard Bottleneck (Concepts Only).
                               Residual stream is removed entirely.
    """
    def __init__(self, input_dim, concept_dim, hidden_dim=4096, use_adversarial=False):
        super().__init__()
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
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            nn.Linear(hidden_dim, input_dim)
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
        # Flatten input (Batch, Seq, Dim) -> (Batch, Seq*Dim) if needed
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        # 1. Encode -> Latent z
        z = self.encoder(x_flat)
        
        # 2. Split Concepts (c) and Residual (u)
        if self.unsupervised_dim > 0:
            c = z[:, :-self.unsupervised_dim] 
            u = z[:, -self.unsupervised_dim:]
        else:
            # Hard Bottleneck: The entire z is the concept vector
            c = z
            u = None 
        
        # 3. Adversarial Pass (if enabled)
        adv_logits = None
        if self.use_adversarial and u is not None:
            # GRL reverses gradient here to confuse the classifier
            u_reversed = self.grl(u)
            adv_logits = self.residual_classifier(u_reversed)
        
        # 4. Decode
        recon = self.decoder(z)
        
        # Reshape back to original input for the Transformer
        if len(original_shape) > 2:
            recon = recon.view(original_shape)
            
        if self.use_adversarial:
            return recon, c, adv_logits
        return recon, c

# ==========================================
# 3. Stream Implementations (Text & Image)
# ==========================================

class Text_CB_AE(CB_AE_Base):
    """
    Specialized for Text Stream (C_ctxt).
    Uses Mean Pooling to compress the sequence (L=154) into a semantic vector
    before bottlenecking, enforcing global semantic constraints.
    """
    def __init__(self, seq_len=154, embed_dim=4096, concept_dim=20, use_adversarial=False):
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # We pool L=154 -> 1, so input to MLP is just embed_dim
        super().__init__(input_dim=embed_dim, concept_dim=concept_dim, use_adversarial=use_adversarial)

    def forward(self, x):
        # x shape: (Batch, 154, 4096)
        
        # 1. POOLING: Compress sequence to get global semantics
        x_pooled = x.mean(dim=1) # (Batch, 4096)
        
        # 2. Base Forward (Encode -> Bottleneck -> Decode)
        # Note: The decoder reconstructs the POOLED vector (Batch, 4096)
        if self.use_adversarial:
            recon_pooled, c, adv_logits = super().forward(x_pooled)
        else:
            recon_pooled, c = super().forward(x_pooled)
            adv_logits = None
            
        # 3. UNPOOLING / BROADCASTING
        # We must return (Batch, 154, 4096) for the MM-DiT.
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

class Image_CB_AE(CB_AE_Base):
    """
    Specialized for Image Stream (x_t).
    Flattens the image patches to preserve spatial detail.
    
    Input: (Batch, num_patches, patch_dim) e.g., (B, 256, 4096)
    Process: Flatten -> Encode -> Bottleneck -> Decode -> Reshape
    Output: (Batch, num_patches, patch_dim) - reconstructed image patches
    """
    def __init__(self, num_patches=256, patch_dim=4096, concept_dim=20, use_adversarial=False):
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        # We flatten patches, so input is huge (256 * 4096)
        # In practice, you might want a Conv1D encoder here to save params, 
        # but this follows your 'flatten' request.
        super().__init__(input_dim=num_patches*patch_dim, concept_dim=concept_dim, use_adversarial=use_adversarial)
    
    def forward(self, x):
        """
        Forward pass for image patches.
        
        Args:
            x: Image patches, shape (Batch, num_patches, patch_dim)
        
        Returns:
            recon: Reconstructed patches, shape (Batch, num_patches, patch_dim)
            c: Concept logits, shape (Batch, concept_dim)
            adv_logits: (Optional) Adversarial logits if use_adversarial=True
        """
        # x shape: (Batch, num_patches, patch_dim) e.g., (B, 256, 4096)
        original_shape = x.shape
        
        # Base forward handles flattening automatically: (B, L, D) -> (B, L*D)
        # Then encodes to bottleneck, decodes, and we reshape back
        if self.use_adversarial:
            recon_flat, c, adv_logits = super().forward(x)
        else:
            recon_flat, c = super().forward(x)
            adv_logits = None
        
        # Reshape back to (Batch, num_patches, patch_dim)
        # The base class already handles this via view(), but we make it explicit here
        recon = recon_flat.view(original_shape)
        
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
    # Hyperparams
    BS = 4
    SEQ_LEN = 154
    PATCHES = 256
    DIM = 4096
    CONCEPTS = 20
    
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