"""GPT-2 style decoder-only transformer in pure JAX.

Parameters are PyTrees built from NamedTuples so `jax.grad`, `jax.tree.map`,
and `jax.jit` work without custom registration.

Tensor variables use Noam suffix notation for axes:
b batch, s sequence, h hidden, v vocab, p heads, d head dim, i/o matmul axes.
Function names are plain — Noam suffixes appear on array variables only (not int scalars).

Architecture notes vs Radford et al. (2019):
  - Demo config: GPT-2 Small — 12 layers, 768 hidden, 12 heads, ctx 1024 (~124M params).
  - Matches: pre-LN blocks, learned position embeddings, causal self-attention,
    GELU MLP (4× hidden), weight-tied token embedding / LM head.
  - Differs: separate Q/K/V projections (paper fuses them); no dropout; no
    residual-path init scaling (1/√N); demo uses plain SGD not Adam+warmup.
  - Tokenizer: not part of the model — it maps text ↔ integer ids. The demo
    uses GPT-2 byte-level BPE via `tiktoken` (vocab 50,257, same as the paper).
"""

from __future__ import annotations

import os

# Silence XLA GPU autotuning warnings (must be set before importing JAX).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import jax.random as jr
import tiktoken
from jax import Array


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GPTConfig(NamedTuple):
    """GPT-2 Small by default (124M params at vocab 50,257)."""

    n_vocab: int = 50_257
    n_ctx: int = 1024
    n_layer: int = 12
    n_heads: int = 12
    n_hidden: int = 768
    mlp_mult: int = 4


def gpt2_small(n_vocab: int = 50_257) -> GPTConfig:
    """Radford et al. GPT-2 Small: L=12, H=768, 12 heads, ctx=1024."""
    return GPTConfig(n_vocab=n_vocab, n_ctx=1024, n_layer=12, n_heads=12, n_hidden=768)


# ---------------------------------------------------------------------------
# Parameter PyTrees (NamedTuples)
# ---------------------------------------------------------------------------


class LayerNormParams(NamedTuple):
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
    ln_final_h: LayerNormParams


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _linear(key: Array, n_in: int, n_out: int, std: float = 0.02) -> LinearParams:
    # GPT-2 uses N(0, std²) with std=0.02 for all linear weights — a fixed scale
    # that works across hidden dims (~768–1600) and is close to 1/√fan_in.
    w_io = jr.normal(key, (n_in, n_out)) * std
    b_o = jnp.zeros(n_out)
    return LinearParams(w_io=w_io, b_o=b_o)


def _layer_norm(key: Array, n_hidden: int) -> LayerNormParams:
    del key
    return LayerNormParams(gamma_h=jnp.ones(n_hidden), beta_h=jnp.zeros(n_hidden))


def _mha_params(key: Array, n_hidden: int) -> MultiHeadAttentionParams:
    key_q, key_k, key_v, key_o = jr.split(key, 4)
    return MultiHeadAttentionParams(
        proj_q=_linear(key_q, n_hidden, n_hidden),
        proj_k=_linear(key_k, n_hidden, n_hidden),
        proj_v=_linear(key_v, n_hidden, n_hidden),
        proj_o=_linear(key_o, n_hidden, n_hidden),
    )


def _mlp_params(key: Array, n_hidden: int, mlp_mult: int) -> MLPParams:
    n_ff = mlp_mult * n_hidden
    key_up, key_down = jr.split(key)
    return MLPParams(
        fc_up=_linear(key_up, n_hidden, n_ff),
        fc_down=_linear(key_down, n_ff, n_hidden),
    )


def _block_params(key: Array, n_hidden: int, mlp_mult: int) -> BlockParams:
    key_ln1, key_attn, key_ln2, key_mlp = jr.split(key, 4)
    return BlockParams(
        ln1_h=_layer_norm(key_ln1, n_hidden),
        attn=_mha_params(key_attn, n_hidden),
        ln2_h=_layer_norm(key_ln2, n_hidden),
        mlp=_mlp_params(key_mlp, n_hidden, mlp_mult),
    )


def init_params(key: Array, config: GPTConfig) -> TransformerParams:
    key_embed, key_pos, key_blocks = jr.split(key, 3)
    block_keys = jr.split(key_blocks, config.n_layer)
    blocks_l = tuple(
        _block_params(k, config.n_hidden, config.mlp_mult) for k in block_keys
    )
    return TransformerParams(
        embed_vh=jr.normal(key_embed, (config.n_vocab, config.n_hidden)) * 0.02,
        pos_embed_sh=jr.normal(key_pos, (config.n_ctx, config.n_hidden)) * 0.01,
        blocks_l=blocks_l,
        ln_final_h=_layer_norm(key_blocks, config.n_hidden),
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def layer_norm(x_bsh: Array, params: LayerNormParams) -> Array:
    mean_h = jnp.mean(x_bsh, axis=-1, keepdims=True)
    var_h = jnp.var(x_bsh, axis=-1, keepdims=True)
    x_hat_bsh = (x_bsh - mean_h) / jnp.sqrt(var_h + 1e-5)
    return params.gamma_h * x_hat_bsh + params.beta_h


def linear(x_bsh: Array, params: LinearParams) -> Array:
    return x_bsh @ params.w_io + params.b_o


def causal_mask(n_seq: int) -> Array:
    return jnp.tril(jnp.ones((n_seq, n_seq)))


def mha(
    x_bsh: Array,
    params: MultiHeadAttentionParams,
    *,
    n_heads: int,
    mask_ss: Array,
) -> Array:
    n_batch, n_seq, n_hidden = x_bsh.shape
    n_head_dim = n_hidden // n_heads

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
    out_bpsd = attn_bpss @ v_bpsd  # (B, P, S, D)

    out_bsh = jnp.transpose(out_bpsd, (0, 2, 1, 3)).reshape(n_batch, n_seq, n_hidden)
    return linear(out_bsh, params.proj_o)


def mlp(x_bsh: Array, params: MLPParams) -> Array:
    x_bsh = gelu(linear(x_bsh, params.fc_up))
    return linear(x_bsh, params.fc_down)


def block(
    x_bsh: Array,
    params: BlockParams,
    *,
    n_heads: int,
    mask_ss: Array,
) -> Array:
    x_bsh = x_bsh + mha(layer_norm(x_bsh, params.ln1_h), params.attn, n_heads=n_heads, mask_ss=mask_ss)
    x_bsh = x_bsh + mlp(layer_norm(x_bsh, params.ln2_h), params.mlp)
    return x_bsh


def forward(
    params: TransformerParams,
    tokens_bs: Array,
    *,
    config: GPTConfig,
) -> Array:
    """Return logits_bsv of shape (B, S, V)."""
    n_batch, n_seq = tokens_bs.shape
    if n_seq > config.n_ctx:
        raise ValueError(f"sequence length {n_seq} exceeds n_ctx {config.n_ctx}")

    pos_s = jnp.arange(n_seq)
    x_bsh = params.embed_vh[tokens_bs] + params.pos_embed_sh[pos_s]  # (B, S, H)
    mask_ss = causal_mask(n_seq)

    for block_params in params.blocks_l:
        x_bsh = block(x_bsh, block_params, n_heads=config.n_heads, mask_ss=mask_ss)

    x_bsh = layer_norm(x_bsh, params.ln_final_h)
    return x_bsh @ params.embed_vh.T  # weight-tied LM head → (B, S, V)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def cross_entropy_loss(logits_bsv: Array, targets_bs: Array) -> Array:
    log_probs_bsv = jax.nn.log_softmax(logits_bsv, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs_bsv, targets_bs[..., None], axis=-1))


def loss(
    params: TransformerParams,
    tokens_bs: Array,
    targets_bs: Array,
    config: GPTConfig,
) -> Array:
    return cross_entropy_loss(forward(params, tokens_bs, config=config), targets_bs)


# ---------------------------------------------------------------------------
# Optimiser — fixed learning rate, no optimizer state
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-3
# GPT-2 Small on one GPU: batch 64 × seq 1024 (65,536 tokens/step); paper used 512
# sequences with distributed training. ~3k steps ≈ 15 min steady-state on GB200.
DEMO_BATCH = 64
DEMO_TRAIN_STEPS = 3_000
DEMO_LOG_EVERY = 100


def sgd_update(params: TransformerParams, grads: TransformerParams) -> TransformerParams:
    return jax.tree.map(lambda p, g: p - LEARNING_RATE * g, params, grads)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def sample_batch(
    key: Array,
    n_batch: int,
    n_seq: int,
    n_vocab: int,
) -> tuple[Array, Array]:
    tokens_bs1 = jr.randint(key, (n_batch, n_seq + 1), 0, n_vocab)
    return tokens_bs1[:, :-1], tokens_bs1[:, 1:]


def train_step(
    params: TransformerParams,
    tokens_bs: Array,
    targets_bs: Array,
    config: GPTConfig,
) -> tuple[TransformerParams, Array]:
    loss_val, grads = jax.value_and_grad(loss)(params, tokens_bs, targets_bs, config)
    params = sgd_update(params, grads)
    return params, loss_val


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def sample_tokens(
    params: TransformerParams,
    prompt_s: Array,
    *,
    config: GPTConfig,
    n_new: int,
    key: Array,
) -> Array:
    """Autoregressive sampling; prompt_s has shape (S,) or (1, S)."""
    if prompt_s.ndim == 1:
        prompt_s = prompt_s[None, :]

    tokens_bs = prompt_s
    for _ in range(n_new):
        ctx_bs = tokens_bs[:, -config.n_ctx :]
        logits_bsv = forward(params, ctx_bs, config=config)
        next_logits_bv = logits_bsv[:, -1, :]
        key, subkey = jr.split(key)
        next_id_b = jr.categorical(subkey, next_logits_bv)
        tokens_bs = jnp.concatenate([tokens_bs, next_id_b[:, None]], axis=1)
    return tokens_bs[0]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    tiny_shakespeare_url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    corpus_path = Path(__file__).resolve().parent / "data" / "tiny_shakespeare.txt"
    if not corpus_path.exists():
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text(urlopen(tiny_shakespeare_url).read().decode(), encoding="utf-8")

    text = corpus_path.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("gpt2")  # GPT-2 paper BPE, vocab 50_257
    corpus_n = jnp.array(enc.encode(text), dtype=jnp.int32)
    print(f"tiny shakespeare: {len(text):,} chars → {corpus_n.shape[0]:,} gpt-2 tokens")

    config = gpt2_small(n_vocab=enc.n_vocab)
    key = jr.key(0)
    key, key_init = jr.split(key, 2)

    params = init_params(key_init, config)
    n_params = sum(leaf.size for leaf in jax.tree.leaves(params))
    print(f"GPT-2 Small: {n_params / 1e6:.1f}M params, {len(jax.tree.leaves(params))} leaves")

    n_batch = DEMO_BATCH
    n_seq = config.n_ctx

    def sample_batch_from_corpus(key: Array) -> tuple[Array, Array]:
        n_tokens = corpus_n.shape[0]
        key, subkey = jr.split(key)
        starts_b = jr.randint(subkey, (n_batch,), 0, n_tokens - n_seq - 1)
        offsets_s1 = jnp.arange(n_seq + 1)
        idx_bs1 = starts_b[:, None] + offsets_s1[None, :]
        chunks_bs1 = corpus_n[idx_bs1]
        return chunks_bs1[:, :-1], chunks_bs1[:, 1:]

    step_fn = jax.jit(lambda p, x, y: train_step(p, x, y, config))
    print(f"training {DEMO_TRAIN_STEPS} steps (batch={n_batch}, seq={n_seq}, "
          f"{n_batch * n_seq:,} tokens/step) …")
    for step in range(DEMO_TRAIN_STEPS):
        key, key_batch = jr.split(key)
        tokens_bs, targets_bs = sample_batch_from_corpus(key_batch)
        params, loss_val = step_fn(params, tokens_bs, targets_bs)
        if step == 0 or (step + 1) % DEMO_LOG_EVERY == 0 or step + 1 == DEMO_TRAIN_STEPS:
            print(f"  step {step + 1:5d}  loss {float(loss_val):.4f}")

    prompt_text = "First Citizen:\nBefore we proceed any further, hear me speak."
    prompt_s = jnp.array(enc.encode(prompt_text), dtype=jnp.int32)
    key, key_gen = jr.split(key)
    out_s = sample_tokens(params, prompt_s, config=config, n_new=40, key=key_gen)
    print(f"\nprompt: {prompt_text!r}")
    print(f"sample: {enc.decode(out_s.tolist())!r}")


if __name__ == "__main__":
    _demo()
