#!/usr/bin/env python3
"""Generate ep06/GPT2_readable.ipynb."""

from __future__ import annotations

import json
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


def main() -> None:
    cells = [
        md(
            """# GPT-2 (Readable) — Flax NNX

Same **decoder-only GPT-2** as [`solution.ipynb`](./solution.ipynb) and [`transformer.py`](../transformer.py), rewritten with **[Flax NNX](https://flax.readthedocs.io/en/latest/why.html)** so the model reads like ordinary Python:

- **`nnx.Module` subclasses** instead of manual `NamedTuple` params
- **`nnx.Linear` / `nnx.LayerNorm` / `nnx.MultiHeadAttention`** instead of `_linear` helpers
- **`nnx.Optimizer` + `@nnx.jit`** instead of hand-rolled `jax.tree.map` SGD

Architecture matches the pure-JAX version: pre-LN blocks, GPT-2 GELU, causal self-attention, weight-tied token embedding / LM head, `std=0.02` linear init (pos embed `0.01`). The demo uses the same **tiny config** and tiny-Shakespeare loop as the instructor notebook.
"""
        ),
        md(
            """## Imports

Set `TF_CPP_MIN_LOG_LEVEL` before JAX (same as the pure-JAX episode).
"""
        ),
        code(
            """import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tiktoken
from flax import nnx
from jax import Array"""
        ),
        md(
            """## Config

Same fields as the pure-JAX episode — `n_` prefix only when it means *number of* (`n_layers`, `n_heads`).
"""
        ),
        code(
            """@dataclass(frozen=True)
class GPTConfig:
    vocab_dim: int = 50_257
    max_ctx: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    hidden: int = 768
    mlp_mult: int = 4


def gpt2_small(vocab_dim: int = 50_257) -> GPTConfig:
    return GPTConfig(vocab_dim=vocab_dim, max_ctx=1024, n_layers=12, n_heads=12, hidden=768)


def demo_config(vocab_dim: int) -> GPTConfig:
    \"\"\"Tiny config for fast notebook demos (matches solution.ipynb).\"\"\"
    return GPTConfig(vocab_dim=vocab_dim, max_ctx=128, n_layers=2, n_heads=4, hidden=64, mlp_mult=4)"""
        ),
        md(
            """## Model

Three small classes mirror the structure in `transformer.py`:

| Pure JAX | NNX here |
|----------|----------|
| `BlockParams` + `block()` | `GPT2Block` |
| `TransformerParams` + `forward()` | `GPT2` |
| `init_params(key, config)` | `GPT2(..., rngs=nnx.Rngs(seed))` |

Weight tying: logits = `x @ token_embed.embedding.T` (same as `x @ embed_vh.T` in pure JAX).
"""
        ),
        code(
            """# GPT-2 weight init (Radford et al.)
INIT_LINEAR = nnx.initializers.normal(stddev=0.02)


def gpt2_gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def causal_mask(n_seq: int) -> Array:
    return jnp.tril(jnp.ones((n_seq, n_seq)))


class GPT2Block(nnx.Module):
    \"\"\"Pre-LN block: LN -> MHA -> residual, LN -> MLP -> residual.\"\"\"

    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        h = config.hidden
        ff = config.mlp_mult * h
        self.ln1 = nnx.LayerNorm(h, epsilon=1e-5, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.n_heads,
            in_features=h,
            kernel_init=INIT_LINEAR,
            out_kernel_init=INIT_LINEAR,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(h, epsilon=1e-5, rngs=rngs)
        self.fc_up = nnx.Linear(h, ff, kernel_init=INIT_LINEAR, rngs=rngs)
        self.fc_down = nnx.Linear(ff, h, kernel_init=INIT_LINEAR, rngs=rngs)

    def __call__(self, x_bsh: Array, mask_ss: Array) -> Array:
        x_bsh = x_bsh + self.attn(self.ln1(x_bsh), mask=mask_ss, decode=False)
        x_bsh = x_bsh + self.fc_down(gpt2_gelu(self.fc_up(self.ln2(x_bsh))))
        return x_bsh


class GPT2(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.token_embed = nnx.Embed(
            config.vocab_dim,
            config.hidden,
            embedding_init=INIT_LINEAR,
            rngs=rngs,
        )
        self.pos_embed = nnx.Param(
            jr.normal(rngs.params(), (config.max_ctx, config.hidden)) * 0.01
        )
        self.blocks = [GPT2Block(config, rngs=rngs) for _ in range(config.n_layers)]
        self.ln_f = nnx.LayerNorm(config.hidden, epsilon=1e-5, rngs=rngs)

    def __call__(self, tokens_bs: Array) -> Array:
        n_seq = tokens_bs.shape[1]
        if n_seq > self.config.max_ctx:
            raise ValueError(f"sequence length {n_seq} exceeds max_ctx {self.config.max_ctx}")

        x_bsh = self.token_embed(tokens_bs) + self.pos_embed[:n_seq]
        mask_ss = causal_mask(n_seq)
        for block in self.blocks:
            x_bsh = block(x_bsh, mask_ss)
        x_bsh = self.ln_f(x_bsh)
        return x_bsh @ self.token_embed.embedding.value.T  # weight-tied LM head"""
        ),
        md(
            """## Loss and training step

`nnx.value_and_grad` differentiates w.r.t. module params; `nnx.Optimizer` applies Optax SGD — same learning rate as the pure-JAX notebook (`1e-3`).
"""
        ),
        code(
            """LEARNING_RATE = 1e-3


def cross_entropy_loss(logits_bsv: Array, targets_bs: Array) -> Array:
    log_probs_bsv = jax.nn.log_softmax(logits_bsv, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs_bsv, targets_bs[..., None], axis=-1))


@nnx.jit
def train_step(model: GPT2, optimizer: nnx.Optimizer, tokens_bs: Array, targets_bs: Array):
    def loss_fn(model: GPT2):
        return cross_entropy_loss(model(tokens_bs), targets_bs)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss"""
        ),
        md(
            """## Demo — tiny Shakespeare

Same data path and training hyperparameters as `solution.ipynb` (`batch=8`, `seq=64`, `50` steps).
"""
        ),
        code(
            """corpus_path = Path("../data/tiny_shakespeare.txt")
if not corpus_path.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(urlopen(url).read().decode(), encoding="utf-8")

text = corpus_path.read_text(encoding="utf-8")
enc = tiktoken.get_encoding("gpt2")
corpus_n = jnp.array(enc.encode(text), dtype=jnp.int32)
print(f"corpus: {len(text):,} chars -> {corpus_n.shape[0]:,} tokens")

config = demo_config(vocab_dim=enc.n_vocab)
model = GPT2(config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.sgd(LEARNING_RATE), wrt=nnx.Param)

n_params = sum(leaf.size for leaf in jax.tree.leaves(nnx.state(model)))
print(f"model: {n_params / 1e6:.2f}M params")
print(f"gpt2_small preset: {gpt2_small(vocab_dim=enc.n_vocab)}")"""
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

key = jr.key(1)
print(f"training {N_STEPS} steps (batch={N_BATCH}, seq={N_SEQ}) ...")
for step in range(N_STEPS):
    key, key_batch = jr.split(key)
    tokens_bs, targets_bs = sample_batch_from_corpus(key_batch, corpus_n, N_BATCH, N_SEQ)
    loss_val = train_step(model, optimizer, tokens_bs, targets_bs)
    if step == 0 or (step + 1) % LOG_EVERY == 0 or step + 1 == N_STEPS:
        print(f"  step {step + 1:3d}  loss {float(loss_val):.4f}")"""
        ),
        md(
            """## Generation

Autoregressive sampling — truncate context to `max_ctx`, same as pure JAX `sample_tokens`.
"""
        ),
        code(
            """@nnx.jit
def forward(model: GPT2, tokens_bs: Array) -> Array:
    return model(tokens_bs)


def sample_tokens(
    model: GPT2,
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
        logits_bsv = forward(model, ctx_bs)
        next_logits_bv = logits_bsv[:, -1, :]
        key, subkey = jr.split(key)
        next_id_b = jr.categorical(subkey, next_logits_bv)
        tokens_bs = jnp.concatenate([tokens_bs, next_id_b[:, None]], axis=1)
    return tokens_bs[0]


prompt_text = "First Citizen:\\nBefore we proceed any further, hear me speak."
prompt_s = jnp.array(enc.encode(prompt_text), dtype=jnp.int32)
key, key_gen = jr.split(key)
out_s = sample_tokens(model, prompt_s, config=config, n_new=40, key=key_gen)
print(f"prompt: {prompt_text!r}")
print(f"sample: {enc.decode(out_s.tolist())!r}")"""
        ),
        md(
            """---

**Compare with pure JAX:** open `solution.ipynb` side-by-side — the math is the same; NNX just owns the parameter tree and layer modules for you. For production-scale GPT-2 Small (`gpt2_small()`), swap `demo_config` for `gpt2_small` and scale batch/seq as in `transformer.py`.
"""
        ),
    ]

    path = ROOT / "ep06" / "GPT2_readable.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
