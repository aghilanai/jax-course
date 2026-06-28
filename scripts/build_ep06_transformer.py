#!/usr/bin/env python3
"""Build Episode 6 — GPT-2 transformer in pure JAX (solution + student notebooks)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def impl_md(title: str, body: str, func: str, sig: str, inputs: str, returns: str) -> dict:
    return md(f"""{title}

{body}

### `{func}{sig}`

| | |
|---|---|
{inputs}
| **returns** | {returns} |

**Implement in the code cell below.**
""")


EP06 = [
    md(
        """# Episode 6 — GPT-2 Transformer in Pure JAX

**Instructor notebook** · run top-to-bottom before recording.

Build a **decoder-only GPT-2** transformer from scratch: parameter **PyTrees** (`NamedTuple`), forward pass, cross-entropy loss, and **plain SGD** updates — no Flax, no Optax.

This episode mirrors [`transformer.py`](../transformer.py) in the repo root. Tensor axes use **Noam suffix notation**: `b` batch, `s` sequence, `h` hidden, `v` vocab, `p` heads, `d` head dim, `i`/`o` matmul axes.

| | |
|---|---|
| **Chapter** | 2.1 · Part II — GPT-2 Transformer |
| **Prereq** | Episodes 1–5 (especially pytrees and `grad`) |
| **Next** | Part II — training at scale |

**References:** [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html)
"""
    ),
    md(
        """## Imports and environment

Standard imports for the episode. `TF_CPP_MIN_LOG_LEVEL=2` silences noisy XLA GPU autotuning logs — set it **before** importing JAX.
"""
    ),
    code(
        """import os
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import jax.random as jr
import tiktoken
from jax import Array"""
    ),
    md(
        """## Config

`GPTConfig` holds model hyperparameters. `gpt2_small()` returns the **124M-parameter** GPT-2 Small defaults (12 layers, 768 hidden, 12 heads, ctx 1024).

The demo at the end uses a **tiny** config so the notebook trains in reasonable time on a single GPU.
"""
    ),
    impl_md(
        "### Config types",
        "Define the config NamedTuple, GPT-2 Small preset, and a tiny demo config.",
        "GPTConfig, gpt2_small, demo_config",
        "",
        "| **`GPTConfig`** | fields: `vocab_dim`, `max_ctx`, `n_layers`, `n_heads`, `hidden`, `mlp_mult` |\n| **`gpt2_small(vocab_dim=50257)`** | returns 124M-param GPT-2 Small defaults |\n| **`demo_config(vocab_dim)`** | returns tiny config (2 layers, 64 hidden) for notebook demos |",
        "`GPTConfig` NamedTuple; two helper functions returning `GPTConfig`",
    ),
    code(
        """class GPTConfig(NamedTuple):
    \"\"\"GPT-2 Small by default (124M params at vocab 50,257).\"\"\"

    vocab_dim: int = 50_257
    max_ctx: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    hidden: int = 768
    mlp_mult: int = 4


def gpt2_small(vocab_dim: int = 50_257) -> GPTConfig:
    \"\"\"Radford et al. GPT-2 Small: L=12, H=768, 12 heads, ctx=1024.\"\"\"
    return GPTConfig(vocab_dim=vocab_dim, max_ctx=1024, n_layers=12, n_heads=12, hidden=768, mlp_mult=4)


def demo_config(vocab_dim: int) -> GPTConfig:
    \"\"\"Small config for notebook demos (fast to init and train).\"\"\"
    return GPTConfig(vocab_dim=vocab_dim, max_ctx=128, n_layers=2, n_heads=4, hidden=64, mlp_mult=4)"""
    ),
    md(
        """## Parameter PyTrees

All learnable weights live in **NamedTuples** so `jax.grad`, `jax.tree.map`, and `jax.jit` work without custom registration (Episode 5).

| Type | Fields | Shapes |
|------|--------|--------|
| `LayerNormParams` | `gamma_h`, `beta_h` | `(H,)`, `(H,)` |
| `LinearParams` | `w_io`, `b_o` | `(I, O)`, `(O,)` |
| `MultiHeadAttentionParams` | `proj_q/k/v/o` | each `LinearParams (H, H)` |
| `MLPParams` | `fc_up`, `fc_down` | `(H, 4H)`, `(4H, H)` |
| `BlockParams` | `ln1_h`, `attn`, `ln2_h`, `mlp` | one pre-LN block |
| `TransformerParams` | `embed_vh`, `pos_embed_sh`, `blocks_l`, `ln_final_h` | token + position embeds, stack of blocks |
"""
    ),
    impl_md(
        "### Parameter types",
        "Declare every parameter container as a `NamedTuple`. No methods — just typed fields.",
        "LayerNormParams, LinearParams, MultiHeadAttentionParams, MLPParams, BlockParams, TransformerParams",
        "",
        "| *(fields)* | see shape table above |",
        "the six NamedTuple classes",
    ),
    code(
        """class LayerNormParams(NamedTuple):
    gamma_h: Array  # (H,)
    beta_h: Array  # (H,)


class LinearParams(NamedTuple):
    w_io: Array  # (I, O)
    b_o: Array  # (O,)


class MultiHeadAttentionParams(NamedTuple):
    proj_q: LinearParams  # (H, H)
    proj_k: LinearParams  # (H, H)
    proj_v: LinearParams  # (H, H)
    proj_o: LinearParams  # (H, H)


class MLPParams(NamedTuple):
    fc_up: LinearParams  # (H, mlp_mult * H)
    fc_down: LinearParams  # (mlp_mult * H, H)


class BlockParams(NamedTuple):
    ln1_h: LayerNormParams
    attn: MultiHeadAttentionParams
    ln2_h: LayerNormParams
    mlp: MLPParams


class TransformerParams(NamedTuple):
    embed_vh: Array  # (V, H) token embeddings — tied to LM head
    pos_embed_sh: Array  # (S, H) position embeddings
    blocks_l: tuple[BlockParams, ...]
    ln_final_h: LayerNormParams"""
    ),
    md(
        """## Initialisation

GPT-2 style init: linear weights \(\mathcal{N}(0, 0.02^2)\), token embeddings `* 0.02`, position embeddings `* 0.01`, layer-norm `gamma=1` / `beta=0`, linear biases `0`.

Split PRNG keys per submodule (Episode 1).
"""
    ),
    impl_md(
        "### Linear weight init",
        "Sample Gaussian weights and zero biases.",
        "_linear",
        "(key: Array, n_in: int, n_out: int, std: float = 0.02) -> LinearParams",
        "| **`key`** | PRNG key |\n| **`n_in`** | input features `I` |\n| **`n_out`** | output features `O` |\n| **`std`** | Gaussian scale (default `0.02`) |",
        "`LinearParams(w_io=(I, O), b_o=(O,))`",
    ),
    code(
        """def _linear(key: Array, n_in: int, n_out: int, std: float = 0.02) -> LinearParams:
    w_io = jr.normal(key, (n_in, n_out)) * std
    b_o = jnp.zeros(n_out)
    return LinearParams(w_io=w_io, b_o=b_o)"""
    ),
    impl_md(
        "### LayerNorm param init",
        "Identity scale, zero shift at start.",
        "_layer_norm",
        "(key: Array, hidden: int) -> LayerNormParams",
        "| **`key`** | unused (API symmetry with other inits) |\n| **`hidden`** | hidden size `H` |",
        "`LayerNormParams(gamma_h=(H,), beta_h=(H,))`",
    ),
    code(
        """def _layer_norm(key: Array, hidden: int) -> LayerNormParams:
    del key
    return LayerNormParams(gamma_h=jnp.ones(hidden), beta_h=jnp.zeros(hidden))"""
    ),
    impl_md(
        "### Multi-head attention params",
        "Four separate Q/K/V/O linear projections.",
        "_mha_params",
        "(key: Array, hidden: int) -> MultiHeadAttentionParams",
        "| **`key`** | split into 4 subkeys |\n| **`hidden`** | model width `H` |",
        "`MultiHeadAttentionParams` with four `LinearParams(H, H)`",
    ),
    code(
        """def _mha_params(key: Array, hidden: int) -> MultiHeadAttentionParams:
    key_q, key_k, key_v, key_o = jr.split(key, 4)
    return MultiHeadAttentionParams(
        proj_q=_linear(key_q, hidden, hidden),
        proj_k=_linear(key_k, hidden, hidden),
        proj_v=_linear(key_v, hidden, hidden),
        proj_o=_linear(key_o, hidden, hidden),
    )"""
    ),
    impl_md(
        "### MLP block params",
        "Up-project with GELU, then down-project back to hidden size.",
        "_mlp_params",
        "(key: Array, hidden: int, mlp_mult: int) -> MLPParams",
        "| **`key`** | split into 2 subkeys |\n| **`hidden`** | `H` |\n| **`mlp_mult`** | expansion factor (4 for GPT-2) |",
        "`MLPParams(fc_up=(H, mlp_mult*H), fc_down=(mlp_mult*H, H))`",
    ),
    code(
        """def _mlp_params(key: Array, hidden: int, mlp_mult: int) -> MLPParams:
    n_ff = mlp_mult * hidden
    key_up, key_down = jr.split(key)
    return MLPParams(
        fc_up=_linear(key_up, hidden, n_ff),
        fc_down=_linear(key_down, n_ff, hidden),
    )"""
    ),
    impl_md(
        "### One transformer block params",
        "Pre-LN block: LN → attention → residual, LN → MLP → residual.",
        "_block_params",
        "(key: Array, hidden: int, mlp_mult: int) -> BlockParams",
        "| **`key`** | split into 4 subkeys (ln1, attn, ln2, mlp) |\n| **`hidden`** | `H` |\n| **`mlp_mult`** | MLP expansion |",
        "`BlockParams`",
    ),
    code(
        """def _block_params(key: Array, hidden: int, mlp_mult: int) -> BlockParams:
    key_ln1, key_attn, key_ln2, key_mlp = jr.split(key, 4)
    return BlockParams(
        ln1_h=_layer_norm(key_ln1, hidden),
        attn=_mha_params(key_attn, hidden),
        ln2_h=_layer_norm(key_ln2, hidden),
        mlp=_mlp_params(key_mlp, hidden, mlp_mult),
    )"""
    ),
    impl_md(
        "### Full model init",
        "Token + position embeddings, `n_layers` blocks, final layer norm.",
        "init_params",
        "(key: Array, config: GPTConfig) -> TransformerParams",
        "| **`key`** | split for embed / pos / blocks |\n| **`config`** | `GPTConfig` |",
        "`TransformerParams`",
    ),
    code(
        """def init_params(key: Array, config: GPTConfig) -> TransformerParams:
    key_embed, key_pos, key_blocks = jr.split(key, 3)
    block_keys = jr.split(key_blocks, config.n_layers)
    blocks_l = tuple(
        _block_params(k, config.hidden, config.mlp_mult) for k in block_keys
    )
    return TransformerParams(
        embed_vh=jr.normal(key_embed, (config.vocab_dim, config.hidden)) * 0.02,
        pos_embed_sh=jr.normal(key_pos, (config.max_ctx, config.hidden)) * 0.01,
        blocks_l=blocks_l,
        ln_final_h=_layer_norm(key_blocks, config.hidden),
    )"""
    ),
    md(
        """## Forward pass — building blocks

Pre-LN GPT-2 block: layer norm before attention and MLP; residual connections wrap each sublayer.

We'll implement bottom-up: activations → linear → layer norm → attention → MLP → full model.
"""
    ),
    impl_md(
        "### GELU activation",
        "GPT-2 uses the tanh approximation of GELU.",
        "gelu",
        "(x: Array) -> Array",
        "| **`x`** | any-shaped array |",
        "same shape as `x`",
    ),
    code(
        """def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))"""
    ),
    impl_md(
        "### Layer normalization (forward)",
        "Normalize over the last axis; apply learned scale and shift.",
        "layer_norm",
        "(x_bsh: Array, params: LayerNormParams) -> Array",
        "| **`x_bsh`** | `(B, S, H)` activations |\n| **`params`** | `LayerNormParams` |",
        "`(B, S, H)` normalized activations",
    ),
    code(
        """def layer_norm(x_bsh: Array, params: LayerNormParams) -> Array:
    mean_h = jnp.mean(x_bsh, axis=-1, keepdims=True)
    var_h = jnp.var(x_bsh, axis=-1, keepdims=True)
    x_hat_bsh = (x_bsh - mean_h) / jnp.sqrt(var_h + 1e-5)
    return params.gamma_h * x_hat_bsh + params.beta_h"""
    ),
    impl_md(
        "### Affine linear layer (forward)",
        "Matrix multiply plus bias: `y = x @ w + b`.",
        "linear",
        "(x_bsh: Array, params: LinearParams) -> Array",
        "| **`x_bsh`** | `(B, S, I)` |\n| **`params.w_io`** | `(I, O)` |\n| **`params.b_o`** | `(O,)` |",
        "`(B, S, O)`",
    ),
    code(
        """def linear(x_bsh: Array, params: LinearParams) -> Array:
    return x_bsh @ params.w_io + params.b_o"""
    ),
    impl_md(
        "### Causal attention mask",
        "Lower-triangular mask so position `t` cannot attend to future tokens.",
        "causal_mask",
        "(n_seq: int) -> Array",
        "| **`n_seq`** | sequence length `S` |",
        "`(S, S)` float mask (1 = keep, 0 = mask out)",
    ),
    code(
        """def causal_mask(n_seq: int) -> Array:
    return jnp.tril(jnp.ones((n_seq, n_seq)))"""
    ),
    md(
        """### Multi-head self-attention

1. Linear projections → Q, K, V each `(B, S, H)`
2. Reshape to `(B, P, S, D)` with `D = H // P`
3. Scaled dot-product attention with causal mask
4. Merge heads → `(B, S, H)` → output projection

**Shapes through attention:** `scores_bpss (B, P, S, S)`, `attn_bpss` same, `out_bpsd (B, P, S, D)`.
"""
    ),
    impl_md(
        "### Multi-head attention",
        "",
        "mha",
        "(x_bsh: Array, params: MultiHeadAttentionParams, *, n_heads: int, mask_ss: Array) -> Array",
        "| **`x_bsh`** | `(B, S, H)` |\n| **`params`** | Q/K/V/O linear params |\n| **`n_heads`** | `P` (must divide `H`) |\n| **`mask_ss`** | `(S, S)` causal mask |",
        "`(B, S, H)`",
    ),
    code(
        """def mha(
    x_bsh: Array,
    params: MultiHeadAttentionParams,
    *,
    n_heads: int,
    mask_ss: Array,
) -> Array:
    n_batch, n_seq, hidden = x_bsh.shape
    n_head_dim = hidden // n_heads

    q_bsh = linear(x_bsh, params.proj_q)
    k_bsh = linear(x_bsh, params.proj_k)
    v_bsh = linear(x_bsh, params.proj_v)

    def split_heads(z_bsh: Array) -> Array:
        z_bpsh = z_bsh.reshape(n_batch, n_seq, n_heads, n_head_dim)
        return jnp.transpose(z_bpsh, (0, 2, 1, 3))  # (B, P, S, D)

    q_bpsd = split_heads(q_bsh)
    k_bpsd = split_heads(k_bsh)
    v_bpsd = split_heads(v_bsh)

    scores_bpss = (q_bpsd @ jnp.swapaxes(k_bpsd, -2, -1)) / jnp.sqrt(n_head_dim)
    scores_bpss = jnp.where(mask_ss[None, None, :, :] > 0, scores_bpss, -1e10)
    attn_bpss = jax.nn.softmax(scores_bpss, axis=-1)
    out_bpsd = attn_bpss @ v_bpsd

    out_bsh = jnp.transpose(out_bpsd, (0, 2, 1, 3)).reshape(n_batch, n_seq, hidden)
    return linear(out_bsh, params.proj_o)"""
    ),
    impl_md(
        "### Feed-forward MLP",
        "Two linear layers with GELU in the middle (4× expansion inside the block).",
        "mlp",
        "(x_bsh: Array, params: MLPParams) -> Array",
        "| **`x_bsh`** | `(B, S, H)` |\n| **`params`** | `MLPParams` |",
        "`(B, S, H)`",
    ),
    code(
        """def mlp(x_bsh: Array, params: MLPParams) -> Array:
    x_bsh = gelu(linear(x_bsh, params.fc_up))
    return linear(x_bsh, params.fc_down)"""
    ),
    impl_md(
        "### Transformer block (forward)",
        "Pre-LN residual block: `x + attn(LN(x))`, then `x + mlp(LN(x))`.",
        "block",
        "(x_bsh: Array, params: BlockParams, *, n_heads: int, mask_ss: Array) -> Array",
        "| **`x_bsh`** | `(B, S, H)` |\n| **`params`** | one `BlockParams` |\n| **`n_heads`** | attention head count |\n| **`mask_ss`** | causal mask `(S, S)` |",
        "`(B, S, H)`",
    ),
    code(
        """def block(
    x_bsh: Array,
    params: BlockParams,
    *,
    n_heads: int,
    mask_ss: Array,
) -> Array:
    x_bsh = x_bsh + mha(layer_norm(x_bsh, params.ln1_h), params.attn, n_heads=n_heads, mask_ss=mask_ss)
    x_bsh = x_bsh + mlp(layer_norm(x_bsh, params.ln2_h), params.mlp)
    return x_bsh"""
    ),
    impl_md(
        "### Full forward pass",
        "Embed tokens + positions, run all blocks, final LN, **weight-tied** LM head (`logits = x @ embed.T`).",
        "forward",
        "(params: TransformerParams, tokens_bs: Array, *, config: GPTConfig) -> Array",
        "| **`params`** | full model weights |\n| **`tokens_bs`** | `(B, S)` int token ids |\n| **`config`** | `GPTConfig` (for `n_heads`, `max_ctx`) |",
        "`logits_bsv` with shape `(B, S, V)`",
    ),
    code(
        """def forward(
    params: TransformerParams,
    tokens_bs: Array,
    *,
    config: GPTConfig,
) -> Array:
    n_batch, n_seq = tokens_bs.shape
    if n_seq > config.max_ctx:
        raise ValueError(f"sequence length {n_seq} exceeds max_ctx {config.max_ctx}")

    pos_s = jnp.arange(n_seq)
    x_bsh = params.embed_vh[tokens_bs] + params.pos_embed_sh[pos_s]
    mask_ss = causal_mask(n_seq)

    for block_params in params.blocks_l:
        x_bsh = block(x_bsh, block_params, n_heads=config.n_heads, mask_ss=mask_ss)

    x_bsh = layer_norm(x_bsh, params.ln_final_h)
    return x_bsh @ params.embed_vh.T"""
    ),
    md(
        """## Loss

Next-token prediction: cross-entropy between `logits_bsv` and integer targets `targets_bs`. We use `log_softmax` + `take_along_axis` for numerical stability.
"""
    ),
    impl_md(
        "### Cross-entropy from logits",
        "",
        "cross_entropy_loss",
        "(logits_bsv: Array, targets_bs: Array) -> Array",
        "| **`logits_bsv`** | `(B, S, V)` unnormalized scores |\n| **`targets_bs`** | `(B, S)` int token ids |",
        "scalar mean loss (`Array` rank 0)",
    ),
    code(
        """def cross_entropy_loss(logits_bsv: Array, targets_bs: Array) -> Array:
    log_probs_bsv = jax.nn.log_softmax(logits_bsv, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs_bsv, targets_bs[..., None], axis=-1))"""
    ),
    impl_md(
        "### Model loss wrapper",
        "Forward pass then cross-entropy — this is what `grad` differentiates.",
        "loss",
        "(params: TransformerParams, tokens_bs: Array, targets_bs: Array, config: GPTConfig) -> Array",
        "| **`params`** | model weights |\n| **`tokens_bs`** | `(B, S)` input tokens |\n| **`targets_bs`** | `(B, S)` shifted targets (next token) |\n| **`config`** | `GPTConfig` |",
        "scalar loss",
    ),
    code(
        """def loss(
    params: TransformerParams,
    tokens_bs: Array,
    targets_bs: Array,
    config: GPTConfig,
) -> Array:
    return cross_entropy_loss(forward(params, tokens_bs, config=config), targets_bs)"""
    ),
    md(
        """## Optimizer — plain SGD

No Adam, no momentum — just `params - lr * grads` via `jax.tree.map` (Episode 5).
"""
    ),
    impl_md(
        "### SGD update",
        "Apply the same learning rate to every leaf in the parameter pytree.",
        "sgd_update",
        "(params: TransformerParams, grads: TransformerParams, learning_rate: float) -> TransformerParams",
        "| **`params`** | current weights |\n| **`grads`** | same-structure gradient pytree |\n| **`learning_rate`** | scalar step size |",
        "updated `TransformerParams`",
    ),
    code(
        """LEARNING_RATE = 1e-3


def sgd_update(params: TransformerParams, grads: TransformerParams, learning_rate: float = LEARNING_RATE) -> TransformerParams:
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)"""
    ),
    md(
        """## Training step

One step: sample a batch → `value_and_grad(loss)` → SGD update. We'll `jit` the step in the demo.
"""
    ),
    impl_md(
        "### Random batch (synthetic)",
        "Sample contiguous token windows; inputs are `[:, :-1]`, targets are `[:, 1:]`.",
        "sample_batch",
        "(key: Array, n_batch: int, n_seq: int, vocab_dim: int) -> tuple[Array, Array]",
        "| **`key`** | PRNG key |\n| **`n_batch`** | `B` |\n| **`n_seq`** | `S` |\n| **`vocab_dim`** | `V` |",
        "`(tokens_bs, targets_bs)` each `(B, S)`",
    ),
    code(
        """def sample_batch(
    key: Array,
    n_batch: int,
    n_seq: int,
    vocab_dim: int,
) -> tuple[Array, Array]:
    tokens_bs1 = jr.randint(key, (n_batch, n_seq + 1), 0, vocab_dim)
    return tokens_bs1[:, :-1], tokens_bs1[:, 1:]"""
    ),
    impl_md(
        "### One training step",
        "",
        "train_step",
        "(params: TransformerParams, tokens_bs: Array, targets_bs: Array, config: GPTConfig) -> tuple[TransformerParams, Array]",
        "| **`params`** | current weights |\n| **`tokens_bs`** | `(B, S)` inputs |\n| **`targets_bs`** | `(B, S)` targets |\n| **`config`** | `GPTConfig` |",
        "`(params, loss_val)` — updated params and scalar loss",
    ),
    code(
        """def train_step(
    params: TransformerParams,
    tokens_bs: Array,
    targets_bs: Array,
    config: GPTConfig,
) -> tuple[TransformerParams, Array]:
    loss_val, grads = jax.value_and_grad(loss)(params, tokens_bs, targets_bs, config)
    params = sgd_update(params, grads)
    return params, loss_val"""
    ),
    md(
        """## Generation

Autoregressive sampling: append one token at a time using `jax.random.categorical` on the last-position logits. Context is truncated to `max_ctx`.
"""
    ),
    impl_md(
        "### Autoregressive sampling",
        "",
        "sample_tokens",
        "(params: TransformerParams, prompt_s: Array, *, config: GPTConfig, n_new: int, key: Array) -> Array",
        "| **`params`** | trained weights |\n| **`prompt_s`** | `(S,)` or `(1, S)` prompt token ids |\n| **`config`** | `GPTConfig` |\n| **`n_new`** | tokens to generate |\n| **`key`** | PRNG for sampling |",
        "`(S + n_new,)` token ids (1-D)",
    ),
    code(
        """def sample_tokens(
    params: TransformerParams,
    prompt_s: Array,
    *,
    config: GPTConfig,
    n_new: int,
    key: Array,
) -> Array:
    if prompt_s.ndim == 1:
        prompt_s = prompt_s[None, :]

    tokens_bs = prompt_s
    for _ in range(n_new):
        ctx_bs = tokens_bs[:, -config.max_ctx :]
        logits_bsv = forward(params, ctx_bs, config=config)
        next_logits_bv = logits_bsv[:, -1, :]
        key, subkey = jr.split(key)
        next_id_b = jr.categorical(subkey, next_logits_bv)
        tokens_bs = jnp.concatenate([tokens_bs, next_id_b[:, None]], axis=1)
    return tokens_bs[0]"""
    ),
    md(
        """## Demo — train on Tiny Shakespeare

Load the corpus from [`data/tiny_shakespeare.txt`](../data/tiny_shakespeare.txt), tokenize with GPT-2 BPE (`tiktoken`), init a **tiny** model, run a short SGD loop, then sample text.

*(Solution below.)*
"""
    ),
    code(
        """corpus_path = Path(\"../data/tiny_shakespeare.txt\")
if not corpus_path.exists():
    url = \"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(urlopen(url).read().decode(), encoding=\"utf-8\")

text = corpus_path.read_text(encoding=\"utf-8\")
enc = tiktoken.get_encoding(\"gpt2\")
corpus_n = jnp.array(enc.encode(text), dtype=jnp.int32)
print(f\"corpus: {len(text):,} chars → {corpus_n.shape[0]:,} tokens\")

config = demo_config(vocab_dim=enc.n_vocab)
key = jr.key(0)
key, key_init = jr.split(key, 2)

params = init_params(key_init, config)
n_params = sum(leaf.size for leaf in jax.tree.leaves(params))
print(f\"model: {n_params / 1e6:.2f}M params, {len(jax.tree.leaves(params))} pytree leaves\")
print(f\"gpt2_small reference: {gpt2_small(vocab_dim=enc.n_vocab)}\")"""
    ),
    md(
        """### Corpus batch sampler

Sample random `(B, S+1)` windows from the tokenized corpus; split into inputs and shifted targets.
"""
    ),
    code(
        """def sample_batch_from_corpus(
    key: Array,
    corpus_n: Array,
    n_batch: int,
    n_seq: int,
) -> tuple[Array, Array]:
    n_tokens = corpus_n.shape[0]
    key, subkey = jr.split(key)
    starts_b = jr.randint(subkey, (n_batch,), 0, n_tokens - n_seq - 1)
    offsets_s1 = jnp.arange(n_seq + 1)
    idx_bs1 = starts_b[:, None] + offsets_s1[None, :]
    chunks_bs1 = corpus_n[idx_bs1]
    return chunks_bs1[:, :-1], chunks_bs1[:, 1:]


N_BATCH = 8
N_SEQ = 64
N_STEPS = 50
LOG_EVERY = 10

step_fn = jax.jit(lambda p, x, y: train_step(p, x, y, config))

print(f\"training {N_STEPS} steps (batch={N_BATCH}, seq={N_SEQ}) …\")
for step in range(N_STEPS):
    key, key_batch = jr.split(key)
    tokens_bs, targets_bs = sample_batch_from_corpus(key_batch, corpus_n, N_BATCH, N_SEQ)
    params, loss_val = step_fn(params, tokens_bs, targets_bs)
    if step == 0 or (step + 1) % LOG_EVERY == 0 or step + 1 == N_STEPS:
        print(f\"  step {step + 1:3d}  loss {float(loss_val):.4f}\")"""
    ),
    md(
        """### Sample text

Even a tiny model and 50 steps won't produce coherent Shakespeare — the point is the **full pipeline** runs end-to-end in pure JAX.
"""
    ),
    code(
        """prompt_text = \"First Citizen:\\nBefore we proceed any further, hear me speak.\"
prompt_s = jnp.array(enc.encode(prompt_text), dtype=jnp.int32)
key, key_gen = jr.split(key)
out_s = sample_tokens(params, prompt_s, config=config, n_new=40, key=key_gen)
print(f\"prompt: {prompt_text!r}\")
print(f\"sample: {enc.decode(out_s.tolist())!r}\")"""
    ),
    md(
        """---

## Exercise

1. Print `jax.tree.map(lambda x: x.shape, params.blocks_l[0])` and verify every leaf shape matches the tables above.
2. Change `N_STEPS` to 200 and confirm loss decreases (it may plateau quickly with this tiny model).
3. Implement **`count_params(config: GPTConfig) -> int`** that returns total parameter count **without** calling `init_params` (sum the closed-form sizes for embed, pos, each block, etc.).
4. **Bonus:** add a `@jax.jit`-compiled `forward` and compare step time with `%timeit` before and after (warm up the cache first).

*(Solution below.)*

### Exercise 3 — parameter count

Closed form: token embed `V*H`, pos embed `S*H`, each block has `4*(H*H + H)` for Q/K/V/O plus MLP `H*(4H) + 4H*H`, plus two layer norms `2*H` each, final LN `2*H`.
"""
    ),
    code(
        """def count_params(config: GPTConfig) -> int:
    v, s, h, l, m = config.vocab_dim, config.max_ctx, config.hidden, config.n_layers, config.mlp_mult
    ff = m * h
    per_block = (
        4 * (h * h + h)  # Q, K, V, O linear
        + (h * ff + ff)  # MLP up (weights + bias)
        + (ff * h + h)  # MLP down
        + 4 * h  # two layer norms (gamma + beta)
    )
    return v * h + s * h + l * per_block + 2 * h  # + final LN


assert count_params(config) == n_params
print(f\"count_params matches init: {count_params(config):,}\")"""
    ),
]


def main() -> None:
    ep = ROOT / "ep06"
    ep.mkdir(parents=True, exist_ok=True)
    solution = ep / "solution.ipynb"
    solution.write_text(json.dumps(nb(EP06), indent=1) + "\n")
    print(f"wrote {solution}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_student.py"), "ep06"],
        check=True,
    )


if __name__ == "__main__":
    main()
