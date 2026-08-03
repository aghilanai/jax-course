"""GPT-2 Small decoder in Flax NNX — manual multi-head self-attention.

Episode 8 companion sheet (param + memory accounting):
https://docs.google.com/spreadsheets/d/1iC4j94aJiXy7Co1tflcDwPG3VSPCr_qmFJEv2BR26sE/edit?usp=sharing
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax import Array

INIT_LINEAR = nnx.initializers.normal(stddev=0.02)


@dataclass(frozen=True)
class GPTConfig:
    vocab_dim: int = 50_257
    max_ctx: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    hidden: int = 768
    mlp_mult: int = 4


def gpt2_gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def causal_mask(n_seq: int) -> Array:
    return jnp.tril(jnp.ones((n_seq, n_seq)))


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention — Q/K/V/O projections implemented explicitly."""

    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        h = config.hidden
        self.n_heads = config.n_heads
        self.head_dim = h // config.n_heads
        if h % config.n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")

        self.q_proj = nnx.Linear(h, h, kernel_init=INIT_LINEAR, rngs=rngs)  # [H, H]
        self.k_proj = nnx.Linear(h, h, kernel_init=INIT_LINEAR, rngs=rngs)  # [H, H]
        self.v_proj = nnx.Linear(h, h, kernel_init=INIT_LINEAR, rngs=rngs)  # [H, H]
        self.o_proj = nnx.Linear(h, h, kernel_init=INIT_LINEAR, rngs=rngs) # [H, H]
        # 4H^2

    def __call__(self, x_bsh: Array, mask_ss: Array) -> Array:
        # x_bsh: (B, S, H)
        n_batch, n_seq, n_hidden = x_bsh.shape

        q_bsh = self.q_proj(x_bsh)
        k_bsh = self.k_proj(x_bsh)
        v_bsh = self.v_proj(x_bsh)

        def split_heads(z_bsh: Array) -> Array:
            z_bpsh = z_bsh.reshape(n_batch, n_seq, self.n_heads, self.head_dim)
            return jnp.transpose(z_bpsh, (0, 2, 1, 3))  # (B, P, S, D)

        q_bpsd = split_heads(q_bsh)
        k_bpsd = split_heads(k_bsh)
        v_bpsd = split_heads(v_bsh)

        scores_bpss = (q_bpsd @ jnp.swapaxes(k_bpsd, -2, -1)) / jnp.sqrt(self.head_dim)
        scores_bpss = jnp.where(mask_ss[None, None, :, :] > 0, scores_bpss, -1e10)
        attn_bpss = jax.nn.softmax(scores_bpss, axis=-1)
        out_bpsd = attn_bpss @ v_bpsd  # (B, P, S, D)

        out_bsh = jnp.transpose(out_bpsd, (0, 2, 1, 3)).reshape(n_batch, n_seq, n_hidden)
        return self.o_proj(out_bsh)


class GPT2Block(nnx.Module):
    """Pre-LN block: LN → attn → residual, LN → MLP → residual."""

    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        h = config.hidden
        ff = config.mlp_mult * h
        self.ln1 = nnx.LayerNorm(h, epsilon=1e-5, rngs=rngs)  # 2H
        self.attn = CausalSelfAttention(config, rngs=rngs) # 4H^2
        self.ln2 = nnx.LayerNorm(h, epsilon=1e-5, rngs=rngs) # 2H
        self.fc_up = nnx.Linear(h, ff, kernel_init=INIT_LINEAR, rngs=rngs) # [H, 4H]
        self.fc_down = nnx.Linear(ff, h, kernel_init=INIT_LINEAR, rngs=rngs) # [4H, H]
        # [4H^2 + 8H^2] --> 12H^2

    def __call__(self, x_bsh: Array, mask_ss: Array) -> Array:
        x_bsh = x_bsh + self.attn(self.ln1(x_bsh), mask_ss)
        x_bsh = x_bsh + self.fc_down(gpt2_gelu(self.fc_up(self.ln2(x_bsh))))
        return x_bsh


class GPT2(nnx.Module):
    """Decoder-only GPT-2 with weight-tied token embedding / LM head."""

    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embed = nnx.Embed(
            config.vocab_dim,
            config.hidden,
            embedding_init=INIT_LINEAR,
            rngs=rngs,
        ) # [50_257, H]
        self.pos_embed = nnx.Param(
            jr.normal(rngs.params(), (config.max_ctx, config.hidden)) * 0.01
        ) # [1024, H]
        self.blocks = nnx.List(
            [GPT2Block(config, rngs=rngs) for _ in range(config.n_layers)]
        ) # 12 * 12H^2
        self.ln_f = nnx.LayerNorm(config.hidden, epsilon=1e-5, rngs=rngs) # 2H


        # Total PARAMETERS for GPT2 LLM is 50_257H + 1024H + 144H^2
        # PURELY AS A FUNCTION OF H
        # How to go from params to memory??
        # 2 bytes for BF16 Weights: 2*P
        # 2 bytes for BF16 Gradients: 2*P
        

        # Moment1 and Moment2 and Masters Weights: each in FP32
        # 4*P + 4*P + 4*P = 12P for all optimizer state

        # A single parameter need's 16 bytes of memory to train

        # 50_257H + 1024H + 144H^2 (plugin real value of H), then multiply by 16
        # 124_318_464
        # 124_318_464 x 16 bytes ---> 1,989,095,424 --> 2GB for 124M Param Model.
        # 8x to 1B, 1T param is 8000x as much memory
        # Kimi K3 is 3T params, so 24_000x as much memory needed as GPT2 --> 48TB for Kimi K3?

    def __call__(self, tokens_bs: Array) -> Array:
        n_seq = tokens_bs.shape[1]
        if n_seq > self.config.max_ctx:
            raise ValueError(
                f"sequence length {n_seq} exceeds max_ctx {self.config.max_ctx}"
            )

        x_bsh = self.token_embed(tokens_bs) + self.pos_embed[:n_seq]
        mask_ss = causal_mask(n_seq)
        for block in self.blocks:
            x_bsh = block(x_bsh, mask_ss)
        x_bsh = self.ln_f(x_bsh)
        return x_bsh @ self.token_embed.embedding[:].T  # (B, S, V) weight-tied LM head


def count_params(model: nnx.Module) -> int:
    return sum(int(x.size) for x in jax.tree.leaves(nnx.state(model)))


# ---------------------------------------------------------------------------
# GPT-2 Small (124M params, Radford et al.)
# ---------------------------------------------------------------------------
GPT2_SMALL = GPTConfig(
    vocab_dim=50_257,
    max_ctx=1024,
    n_layers=12,
    n_heads=12,
    hidden=768,
    mlp_mult=4,
)


if __name__ == "__main__":
    model = GPT2(GPT2_SMALL, rngs=nnx.Rngs(0))
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = model(tokens)
    print(f"logits shape: {logits.shape}")
    print(f"params: {count_params(model):,} ({count_params(model) / 1e6:.1f}M)")
    print(f"GPT2_SMALL: {GPT2_SMALL}")
