import torch
import torch.nn as nn
from torch.autograd import Function

# ==========================================
# Utilities & Gradient Reversal
# ==========================================

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


def _init_transformer(m):
    """Init transformer-style weights for stability."""
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ==========================================
# MM-DiT style blocks (AdaLN modulation)
# ==========================================

class AdaLNModulation(nn.Module):
    """Single-layer modulation head producing shift/scale/gate triplets."""

    def __init__(self, cond_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim * 3),
        )

    def forward(self, cond):
        shift, scale, gate = self.net(cond).chunk(3, dim=-1)
        return shift, scale, gate


class FeedForward(nn.Module):
    """Gated GELU feed-forward block."""

    def __init__(self, embed_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MMDiTBlock(nn.Module):
    """
    Simplified MM-DiT block with AdaLN modulation on both attention and MLP paths.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, cond_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(embed_dim, ff_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mod_attn = AdaLNModulation(cond_dim, ff_dim, embed_dim)
        self.mod_ff = AdaLNModulation(cond_dim, ff_dim, embed_dim)

    def forward(self, x, cond):
        shift1, scale1, gate1 = self.mod_attn(cond)
        x_norm = self.ln1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out

        shift2, scale2, gate2 = self.mod_ff(cond)
        x_norm2 = self.ln2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.ff(x_norm2)
        return x


# ==========================================
# Unified Transformer CBM (text + image together)
# ==========================================

class ConceptBottleneckTransformer(nn.Module):
    """
    Unified CBM with MM-DiT-style AdaLN blocks for both encoder and decoder.
    Includes optional down-projection to shrink internal model dim for memory savings.
    """

    def __init__(
        self,
        text_len=333,
        image_len=2304,
        embed_dim=1536,
        concept_dim=2,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        ff_dim=2048,
        bottleneck_tokens=2,
        dropout=0.1,
        use_adversarial=False,
        down_proj_dim=512,
    ):
        super().__init__()
        self.text_len = text_len
        self.image_len = image_len
        self.input_dim = embed_dim
        self.model_dim = down_proj_dim or embed_dim
        self.concept_dim = concept_dim
        self.use_adversarial = use_adversarial

        self.unsupervised_dim = 8 if use_adversarial else 0
        self.total_bottleneck = concept_dim + self.unsupervised_dim
        self.num_concept_tokens = bottleneck_tokens

        # Input/output projections to shrink working dimension
        self.input_proj = nn.Linear(self.input_dim, self.model_dim)
        self.output_proj = nn.Linear(self.model_dim, self.input_dim)

        # Token/type/positional embeddings
        self.type_embeddings = nn.Parameter(torch.randn(2, self.model_dim))  # 0=text, 1=image
        self.pos_text = nn.Parameter(torch.randn(1, text_len, self.model_dim))
        self.pos_image = nn.Parameter(torch.randn(1, image_len, self.model_dim))
        self.concept_tokens = nn.Parameter(torch.randn(bottleneck_tokens, self.model_dim))

        # Reconstruction queries (decoded tokens start here)
        self.dec_text_queries = nn.Parameter(torch.randn(text_len, self.model_dim))
        self.dec_image_queries = nn.Parameter(torch.randn(image_len, self.model_dim))
        self.memory_pos = nn.Parameter(torch.randn(1, bottleneck_tokens, self.model_dim))

        # Conditioning projections (use pooled text+image for encoder; bottleneck for decoder)
        self.cond_in = nn.Sequential(
            nn.Linear(self.model_dim * 2, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, self.model_dim),
        )
        self.cond_from_bottleneck = nn.Sequential(
            nn.Linear(self.total_bottleneck, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, self.model_dim),
        )

        # Encoder/decoder stacks of MM-DiT blocks
        self.encoder_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    embed_dim=self.model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    cond_dim=self.model_dim,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    embed_dim=self.model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    cond_dim=self.model_dim,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Bottleneck projections
        self.bottleneck_proj = nn.Linear(self.model_dim, self.total_bottleneck)
        self.memory_proj = nn.Linear(self.total_bottleneck, bottleneck_tokens * self.model_dim)

        if self.use_adversarial:
            self.grl = GradientReversal(alpha=1.0)
            self.residual_classifier = nn.Sequential(
                nn.Linear(self.unsupervised_dim, 64),
                nn.GELU(),
                nn.Linear(64, concept_dim),
            )

        self.apply(_init_transformer)

    def _build_encoder_tokens(self, text_emb, image_lat):
        """Add type + positional encodings and concatenate with concept tokens."""
        batch = text_emb.shape[0]
        text_tokens = text_emb + self.pos_text + self.type_embeddings[0]
        image_tokens = image_lat + self.pos_image + self.type_embeddings[1]

        concept_tokens = self.concept_tokens.unsqueeze(0).expand(batch, -1, -1)
        encoder_input = torch.cat([concept_tokens, text_tokens, image_tokens], dim=1)
        return encoder_input

    def _run_encoder(self, text_emb, image_lat):
        enc_tokens = self._build_encoder_tokens(text_emb, image_lat)
        pooled_text = text_emb.mean(dim=1)
        pooled_img = image_lat.mean(dim=1)
        cond = self.cond_in(torch.cat([pooled_text, pooled_img], dim=-1))

        for blk in self.encoder_blocks:
            enc_tokens = blk(enc_tokens, cond)

        concept_states = enc_tokens[:, : self.num_concept_tokens]
        bottleneck_vec = self.bottleneck_proj(concept_states.mean(dim=1))
        concept_logits = bottleneck_vec[:, : self.concept_dim]
        residual = bottleneck_vec[:, self.concept_dim :] if self.unsupervised_dim > 0 else None
        return enc_tokens, bottleneck_vec, concept_logits, residual

    def _decode_from_bottleneck(self, bottleneck_vec, batch_size):
        """MM-DiT-style decoder conditioned only on bottleneck."""
        cond = self.cond_from_bottleneck(bottleneck_vec)

        memory = self.memory_proj(bottleneck_vec).view(
            batch_size, self.num_concept_tokens, self.model_dim
        )
        memory = memory + self.memory_pos

        text_queries = (
            self.dec_text_queries.unsqueeze(0).expand(batch_size, -1, -1)
            + self.pos_text
            + self.type_embeddings[0]
        )
        image_queries = (
            self.dec_image_queries.unsqueeze(0).expand(batch_size, -1, -1)
            + self.pos_image
            + self.type_embeddings[1]
        )
        dec_tokens = torch.cat([memory, text_queries, image_queries], dim=1)

        for blk in self.decoder_blocks:
            dec_tokens = blk(dec_tokens, cond)

        recon_text = dec_tokens[:, self.num_concept_tokens : self.num_concept_tokens + self.text_len]
        recon_image = dec_tokens[:, self.num_concept_tokens + self.text_len :]
        return recon_text, recon_image

    def forward(self, text_emb, image_lat, force_concept=None):
        """
        Args:
            text_emb: (B, text_len, embed_dim)
            image_lat: (B, image_len, embed_dim)
            force_concept: Optional tensor to override predicted concepts during decoding.
        """
        orig_dtype = text_emb.dtype

        text_emb = self.input_proj(text_emb.float())
        image_lat = self.input_proj(image_lat.float())

        _, bottleneck_vec, concept_logits, residual = self._run_encoder(text_emb, image_lat)

        adv_logits = None
        if self.use_adversarial and residual is not None:
            adv_logits = self.residual_classifier(self.grl(residual))

        bottleneck_for_decode = bottleneck_vec
        if force_concept is not None:
            forced = force_concept.to(bottleneck_vec.device).float()
            if forced.dim() == 1:
                forced = forced.unsqueeze(0).expand(bottleneck_vec.shape[0], -1)
            if forced.shape[1] != self.concept_dim:
                raise ValueError(
                    f"force_concept expected dim {self.concept_dim}, got {forced.shape[1]}"
                )
            if residual is not None:
                bottleneck_for_decode = torch.cat([forced, residual], dim=1)
            else:
                bottleneck_for_decode = forced

        recon_text, recon_image = self._decode_from_bottleneck(
            bottleneck_for_decode, batch_size=text_emb.shape[0]
        )

        recon_text = self.output_proj(recon_text).to(orig_dtype)
        recon_image = self.output_proj(recon_image).to(orig_dtype)

        return (
            recon_text,
            recon_image,
            concept_logits,
            adv_logits,
        )

    def encode_only(self, text_emb, image_lat):
        """Return concept logits (no decode) – used for intervention cycle losses."""
        text_emb = self.input_proj(text_emb.float())
        image_lat = self.input_proj(image_lat.float())
        _, _, concept_logits, _ = self._run_encoder(text_emb, image_lat)
        return concept_logits


# ==========================================
# Objective Function
# ==========================================

class CombinedLoss(nn.Module):
    """
    Composite loss mirroring the paper's objectives with available signals:
      - Lr: latent reconstruction (text/image).
      - Lc: concept alignment with pseudo-labels.
      - Li: intervention consistency (decode with forced concept, re-encode, match target).
      - Ladv: optional adversarial residual confusion.
    """

    def __init__(
        self,
        lambda_recon=1.0,
        lambda_concept=1.0,
        lambda_intervene=1.0,
        lambda_adv=0.5,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_concept = lambda_concept
        self.lambda_intervene = lambda_intervene
        self.lambda_adv = lambda_adv
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, pseudo_labels, intervened=None):
        """
        Args:
            outputs: tuple from ConceptBottleneckTransformer forward
            targets: (target_text, target_image)
            pseudo_labels: ground-truth/multi-hot concept tensor
            intervened: optional dict with keys {"concept_logits": tensor, "target": tensor}
                        produced by decoding with forced concepts then re-encoding.
        """
        recon_text, recon_image, concept_logits, adv_logits = outputs
        target_text, target_image = targets

        loss_recon_text = self.mse(recon_text, target_text)
        loss_recon_image = self.mse(recon_image, target_image)
        loss_recon = loss_recon_text + loss_recon_image

        loss_concept = self.bce(concept_logits, pseudo_labels)

        loss_intervene = torch.tensor(0.0, device=concept_logits.device)
        if intervened is not None:
            loss_intervene = self.bce(
                intervened["concept_logits"],
                intervened["target"],
            )

        loss_adv = torch.tensor(0.0, device=concept_logits.device)
        if adv_logits is not None:
            loss_adv = self.bce(adv_logits, pseudo_labels)

        total = (
            self.lambda_recon * loss_recon
            + self.lambda_concept * loss_concept
            + self.lambda_intervene * loss_intervene
            + self.lambda_adv * loss_adv
        )

        return {
            "total": total,
            "recon": loss_recon.item(),
            "recon_text": loss_recon_text.item(),
            "recon_image": loss_recon_image.item(),
            "concept": loss_concept.item(),
            "intervene": loss_intervene.item(),
            "adversarial": loss_adv.item(),
        }


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    BS = 2
    SEQ_LEN = 333
    PATCHES = 2304
    DIM = 1536
    CONCEPTS = 2

    model = ConceptBottleneckTransformer(
        text_len=SEQ_LEN,
        image_len=PATCHES,
        embed_dim=DIM,
        concept_dim=CONCEPTS,
        use_adversarial=False,
    )
    loss_fn = CombinedLoss()

    txt = torch.randn(BS, SEQ_LEN, DIM)
    img = torch.randn(BS, PATCHES, DIM)
    labels = torch.randint(0, 2, (BS, CONCEPTS)).float()

    outputs = model(txt, img)
    flipped = labels.clone()
    flipped[:, 0] = 1 - flipped[:, 0]  # example intervention
    recon_txt_i, recon_img_i, _, _ = model(txt, img, force_concept=flipped)
    concept_logits_i = model.encode_only(recon_txt_i, recon_img_i)
    losses = loss_fn(
        outputs,
        targets=(txt, img),
        pseudo_labels=labels,
        intervened={"concept_logits": concept_logits_i, "target": flipped},
    )
    losses["total"].backward()

    print("Forward/backward pass complete. Losses:", losses)
