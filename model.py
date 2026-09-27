"""
MiniTruth-8M model

Small Llama-inspired decoder-only transformer.

Architecture:
    Token Embedding
        ↓
    Transformer Blocks
        ├── RMSNorm
        ├── RoPE Self-Attention
        ├── Residual
        ├── RMSNorm
        ├── SwiGLU
        └── Residual
        ↓
    RMSNorm
        ↓
    Language-model head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        # Root mean square
        rms = x.pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            rms + self.eps
        )

        return x * self.weight


# ============================================================
# Rotary positional embeddings
# ============================================================

def build_rope_cache(
    seq_len,
    head_dim,
    theta,
    device,
):
    """
    Precompute cosine and sine values for RoPE.
    """

    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    angles = torch.outer(
        positions,
        frequencies,
    )

    cos = angles.cos()
    sin = angles.sin()

    return cos, sin


def apply_rope(
    x,
    cos,
    sin,
):
    """
    Apply rotary positional embeddings.

    x:
        [batch, heads, sequence, head_dim]
    """

    seq_len = x.size(2)

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # Convert:
    #
    # [cos1, cos2, ...]
    #
    # into:
    #
    # [cos1, cos1, cos2, cos2, ...]

    cos = torch.repeat_interleave(
        cos,
        2,
        dim=-1,
    )

    sin = torch.repeat_interleave(
        sin,
        2,
        dim=-1,
    )

    # Rotate pairs of dimensions.

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    rotated = torch.stack(
        (-x2, x1),
        dim=-1,
    ).flatten(-2)

    return (
        x * cos
        + rotated * sin
    )


# ============================================================
# Causal self-attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0

        self.n_head = cfg.n_head

        self.head_dim = (
            cfg.n_embd // cfg.n_head
        )

        self.n_embd = cfg.n_embd

        # Q, K, V
        self.q_proj = nn.Linear(
            cfg.n_embd,
            cfg.n_embd,
            bias=cfg.bias,
        )

        self.k_proj = nn.Linear(
            cfg.n_embd,
            cfg.n_embd,
            bias=cfg.bias,
        )

        self.v_proj = nn.Linear(
            cfg.n_embd,
            cfg.n_embd,
            bias=cfg.bias,
        )

        self.o_proj = nn.Linear(
            cfg.n_embd,
            cfg.n_embd,
            bias=cfg.bias,
        )

        # RoPE cache
        cos, sin = build_rope_cache(
            cfg.block_size,
            self.head_dim,
            cfg.rope_theta,
            device="cpu",
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False,
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------------------------
        # Reshape into attention heads
        # ----------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Rotary positional embeddings
        # ----------------------------------------------------

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin,
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            batch_size,
            seq_len,
            self.n_embd,
        )

        return self.o_proj(y)


# ============================================================
# SwiGLU feed-forward network
# ============================================================

class SwiGLU(nn.Module):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            cfg.n_embd,
            cfg.n_ff,
            bias=cfg.bias,
        )

        self.up_proj = nn.Linear(
            cfg.n_embd,
            cfg.n_ff,
            bias=cfg.bias,
        )

        self.down_proj = nn.Linear(
            cfg.n_ff,
            cfg.n_embd,
            bias=cfg.bias,
        )

    def forward(self, x):

        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        return self.down_proj(
            gate * up
        )


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(
            cfg.n_embd
        )

        self.attention = CausalSelfAttention(
            cfg
        )

        self.ffn_norm = RMSNorm(
            cfg.n_embd
        )

        self.ffn = SwiGLU(
            cfg
        )

    def forward(self, x):

        # Attention residual
        x = x + self.attention(
            self.attn_norm(x)
        )

        # Feed-forward residual
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x


# ============================================================
# MiniTruth-8M
# ============================================================

class MiniTruthLM(nn.Module):

    def __init__(
        self,
        cfg=None,
    ):
        super().__init__()

        if cfg is None:
            cfg = config.model

        self.cfg = cfg

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.n_embd,
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg)
                for _ in range(cfg.n_layer)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_norm = RMSNorm(
            cfg.n_embd
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            cfg.n_embd,
            cfg.vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        if cfg.tie_embeddings:

            self.lm_head.weight = (
                self.token_embedding.weight
            )

        # ----------------------------------------------------
        # Initialize weights
        # ----------------------------------------------------

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module,
    ):

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids,
        targets=None,
    ):

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        x = self.token_embedding(
            input_ids
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final norm
        # ----------------------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------------------
        # Language-model logits
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # ----------------------------------------------------
        # Optional training loss
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1),
                ),
                targets.view(-1),
            )

        return logits, loss


# ============================================================
# Parameter counting
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Creating MiniTruth-8M")
    print("=" * 60)

    model = MiniTruthLM()

    parameters = count_parameters(model)

    print(
        f"\nTrainable parameters: "
        f"{parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{parameters / 1_000_000:.2f}M"
    )

    # --------------------------------------------------------
    # Test forward pass
    # --------------------------------------------------------

    batch_size = 2
    sequence_length = 32

    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    targets = torch.randint(
        0,
        config.model.vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    print(
        f"\nInput shape:  "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Logits shape: "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Loss:         "
        f"{loss.item():.4f}"
    )

    print("\nModel test successful.")