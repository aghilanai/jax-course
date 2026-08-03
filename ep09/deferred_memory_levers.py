"""Episode 9 — Memory levers & mixed precision.

Teaches, in order:
  1. Training memory budget (params / grads / optimizer / activations)
  2. Activation checkpointing — ``jax.checkpoint`` on ``block``
  3. BF16 compute with FP32 master weights + AdamW
  4. Buffer donation — ``donate_argnums`` in ``jit``
  5. Gradient accumulation — large effective batch, small micro-batch

Run::
    python ep09/deferred_memory_levers.py
    python ep09/deferred_memory_levers.py --profile          # XProf Memory tab
    python ep09/deferred_memory_levers.py --find-max-batch   # max batch before OOM

Prereq memory intuition: ``ep08/gpt2.py``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import urlopen

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import jax.random as jr
import tiktoken
from jax import Array

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GPTConfig(NamedTuple):
    vocab_dim: int = 50_257
    max_ctx: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    hidden: int = 768
    mlp_mult: int = 4


def gpt2_small(vocab_dim: int = 50_257) -> GPTConfig:
    return GPTConfig(
        vocab_dim=vocab_dim,
        max_ctx=1024,
        n_layers=12,
        n_heads=12,
        hidden=768,
        mlp_mult=4,
    )


def demo_config(vocab_dim: int) -> GPTConfig:
    """Tiny config for smoke tests."""
    return GPTConfig(
        vocab_dim=vocab_dim,
        max_ctx=128,
        n_layers=2,
        n_heads=4,
        hidden=64,
        mlp_mult=4,
    )


def stress_config(vocab_dim: int) -> GPTConfig:
    """Large enough that memory tricks matter on one GPU (~16M params)."""
    return GPTConfig(
        vocab_dim=vocab_dim,
        max_ctx=256,
        n_layers=6,
        n_heads=8,
        hidden=256,
        mlp_mult=4,
    )


# ---------------------------------------------------------------------------
# Parameter PyTrees (Episode 6 baseline)
# ---------------------------------------------------------------------------


class LayerNormParams(NamedTuple):
    gamma_h: Array
    beta_h: Array


class LinearParams(NamedTuple):
    w_io: Array
    b_o: Array


class MultiHeadAttentionParams(NamedTuple):
    proj_q: LinearParams
    proj_k: LinearParams
    proj_v: LinearParams
    proj_o: LinearParams


class MLPParams(NamedTuple):
    fc_hf: LinearParams
    fc_fh: LinearParams


class BlockParams(NamedTuple):
    ln1_h: LayerNormParams
    attn_hh: MultiHeadAttentionParams
    ln2_h: LayerNormParams
    mlp_hh: MLPParams


class TransformerParams(NamedTuple):
    embed_vh: Array
    pos_embed_sh: Array
    blocks_l: tuple[BlockParams, ...]
    ln_final_h: LayerNormParams


def _linear(key: Array, n_in: int, n_out: int, std: float = 0.02) -> LinearParams:
    return LinearParams(
        w_io=jr.normal(key, (n_in, n_out)) * std,
        b_o=jnp.zeros(n_out),
    )


def _layer_norm(key: Array, hidden: int) -> LayerNormParams:
    del key
    return LayerNormParams(gamma_h=jnp.ones(hidden), beta_h=jnp.zeros(hidden))


def _mha_params(key: Array, hidden: int) -> MultiHeadAttentionParams:
    key_q, key_k, key_v, key_o = jr.split(key, 4)
    return MultiHeadAttentionParams(
        proj_q=_linear(key_q, hidden, hidden),
        proj_k=_linear(key_k, hidden, hidden),
        proj_v=_linear(key_v, hidden, hidden),
        proj_o=_linear(key_o, hidden, hidden),
    )


def _mlp_params(key: Array, hidden: int, mlp_mult: int) -> MLPParams:
    n_ff = mlp_mult * hidden
    key_up, key_down = jr.split(key)
    return MLPParams(
        fc_hf=_linear(key_up, hidden, n_ff),
        fc_fh=_linear(key_down, n_ff, hidden),
    )


def _block_params(key: Array, hidden: int, mlp_mult: int) -> BlockParams:
    key_ln1, key_attn, key_ln2, key_mlp = jr.split(key, 4)
    return BlockParams(
        ln1_h=_layer_norm(key_ln1, hidden),
        attn_hh=_mha_params(key_attn, hidden),
        ln2_h=_layer_norm(key_ln2, hidden),
        mlp_hh=_mlp_params(key_mlp, hidden, mlp_mult),
    )


def init_params(key: Array, config: GPTConfig) -> TransformerParams:
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
    )


def count_params(config: GPTConfig) -> int:
    v, s, h, l, m = (
        config.vocab_dim,
        config.max_ctx,
        config.hidden,
        config.n_layers,
        config.mlp_mult,
    )
    ff = m * h
    per_block = (
        4 * (h * h + h)
        + (h * ff + ff)
        + (ff * h + h)
        + 4 * h
    )
    return v * h + s * h + l * per_block + 2 * h


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def layer_norm(x_bsh: Array, params: LayerNormParams) -> Array:
    x_fp32 = x_bsh.astype(jnp.float32)
    mean_h = jnp.mean(x_fp32, axis=-1, keepdims=True)
    var_h = jnp.var(x_fp32, axis=-1, keepdims=True)
    x_hat = (x_fp32 - mean_h) / jnp.sqrt(var_h + 1e-5)
    return params.gamma_h * x_hat + params.beta_h


def linear(x_bsh: Array, params: LinearParams, *, compute_dtype: jnp.dtype) -> Array:
    w = params.w_io.astype(compute_dtype)
    x = x_bsh.astype(compute_dtype)
    y = x @ w
    return y.astype(jnp.float32) + params.b_o


def causal_mask(n_seq: int) -> Array:
    return jnp.tril(jnp.ones((n_seq, n_seq)))


def mha(
    x_bsh: Array,
    params: MultiHeadAttentionParams,
    *,
    n_heads: int,
    mask_ss: Array,
    compute_dtype: jnp.dtype,
) -> Array:
    n_batch, n_seq, n_hidden = x_bsh.shape
    n_head_dim = n_hidden // n_heads

    q_bsh = linear(x_bsh, params.proj_q, compute_dtype=compute_dtype)
    k_bsh = linear(x_bsh, params.proj_k, compute_dtype=compute_dtype)
    v_bsh = linear(x_bsh, params.proj_v, compute_dtype=compute_dtype)

    def split_heads(z_bsh: Array) -> Array:
        z_bpsh = z_bsh.reshape(n_batch, n_seq, n_heads, n_head_dim)
        return jnp.transpose(z_bpsh, (0, 2, 1, 3))

    q_bpsd = split_heads(q_bsh)
    k_bpsd = split_heads(k_bsh)
    v_bpsd = split_heads(v_bsh)

    scores_bpss = (q_bpsd @ jnp.swapaxes(k_bpsd, -2, -1)) / jnp.sqrt(n_head_dim)
    scores_bpss = jnp.where(mask_ss[None, None, :, :] > 0, scores_bpss, -1e10)
    attn_bpss = jax.nn.softmax(scores_bpss, axis=-1)
    out_bpsd = attn_bpss @ v_bpsd

    out_bsh = jnp.transpose(out_bpsd, (0, 2, 1, 3)).reshape(n_batch, n_seq, n_hidden)
    return linear(out_bsh, params.proj_o, compute_dtype=compute_dtype)


def mlp(x_bsh: Array, params: MLPParams, *, compute_dtype: jnp.dtype) -> Array:
    x_bsh = gelu(linear(x_bsh, params.fc_hf, compute_dtype=compute_dtype))
    return linear(x_bsh, params.fc_fh, compute_dtype=compute_dtype)


def block(
    x_bsh: Array,
    params: BlockParams,
    *,
    n_heads: int,
    mask_ss: Array,
    compute_dtype: jnp.dtype,
) -> Array:
    x_bsh = x_bsh + mha(
        layer_norm(x_bsh, params.ln1_h),
        params.attn_hh,
        n_heads=n_heads,
        mask_ss=mask_ss,
        compute_dtype=compute_dtype,
    )
    x_bsh = x_bsh + mlp(
        layer_norm(x_bsh, params.ln2_h),
        params.mlp_hh,
        compute_dtype=compute_dtype,
    )
    return x_bsh


def forward(
    params: TransformerParams,
    tokens_bs: Array,
    *,
    config: GPTConfig,
    compute_dtype: jnp.dtype = jnp.float32,
    use_checkpoint: bool = False,
) -> Array:
    n_batch, n_seq = tokens_bs.shape
    if n_seq > config.max_ctx:
        raise ValueError(f"sequence length {n_seq} exceeds max_ctx {config.max_ctx}")

    pos_s = jnp.arange(n_seq)
    embed = params.embed_vh.astype(compute_dtype)
    x_bsh = embed[tokens_bs] + params.pos_embed_sh[pos_s].astype(compute_dtype)
    x_bsh = x_bsh.astype(jnp.float32)
    mask_ss = causal_mask(n_seq)

    for block_params in params.blocks_l:
        def make_apply(bp: BlockParams):
            def apply_block(x_bsh: Array) -> Array:
                return block(
                    x_bsh,
                    bp,
                    n_heads=config.n_heads,
                    mask_ss=mask_ss,
                    compute_dtype=compute_dtype,
                )

            return apply_block

        apply_block = make_apply(block_params)
        if use_checkpoint:
            x_bsh = jax.checkpoint(apply_block)(x_bsh)
        else:
            x_bsh = apply_block(x_bsh)

    x_bsh = layer_norm(x_bsh, params.ln_final_h)
    logits = x_bsh @ embed.T
    return logits.astype(jnp.float32)


def cross_entropy_loss(logits_bsv: Array, targets_bs: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits_bsv.astype(jnp.float32), axis=-1)
    return -jnp.mean(
        jnp.take_along_axis(log_probs, targets_bs[..., None], axis=-1)
    )


def loss(
    params: TransformerParams,
    tokens_bs: Array,
    targets_bs: Array,
    config: GPTConfig,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    use_checkpoint: bool = False,
) -> Array:
    logits = forward(
        params,
        tokens_bs,
        config=config,
        compute_dtype=compute_dtype,
        use_checkpoint=use_checkpoint,
    )
    return cross_entropy_loss(logits, targets_bs)


# ---------------------------------------------------------------------------
# §1 Memory budget
# Ref: https://huggingface.co/docs/transformers/perf_train_gpu_one
#      https://blog.eleuther.ai/transformer-math/
# ---------------------------------------------------------------------------


class MeasuredStatic(NamedTuple):
    """Bytes measured by summing PyTree leaves (exact for static tensors)."""

    params_bytes: int | None = None
    grads_bytes: int | None = None
    optimizer_bytes: int | None = None


class MemoryBudget(NamedTuple):
    params_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activation_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.params_bytes
            + self.grads_bytes
            + self.optimizer_bytes
            + self.activation_bytes
        )

    def format(self, measured: MeasuredStatic | None = None) -> str:
        gib = 1024**3

        def line(label: str, est: int, meas: int | None, *, always_est: bool = False) -> str:
            if always_est or meas is None:
                return f"  {label:<20} {est / gib:8.3f} GiB  (est.)"
            return (
                f"  {label:<20} {meas / gib:8.3f} GiB  measured  "
                f"(est {est / gib:.3f})"
            )

        m = measured or MeasuredStatic()
        lines = [
            line("params", self.params_bytes, m.params_bytes),
            line("grads", self.grads_bytes, m.grads_bytes),
            line("optimizer", self.optimizer_bytes, m.optimizer_bytes),
            line(
                "activations",
                self.activation_bytes,
                None,
                always_est=True,
            ),
            "  ─────────────────────────",
        ]
        static_meas = (m.params_bytes or 0) + (m.grads_bytes or 0) + (m.optimizer_bytes or 0)
        static_est = self.params_bytes + self.grads_bytes + self.optimizer_bytes
        if measured and static_meas:
            total_line = (
                f"  {'static subtotal':<20} {static_meas / gib:8.3f} GiB  measured  "
                f"(est {static_est / gib:.3f})"
            )
        else:
            total_line = f"  {'total (est.)':<20} {self.total_bytes / gib:8.3f} GiB"
        lines.append(total_line)
        if measured and static_meas:
            lines.append(
                f"  {'+ activations (est.)':<20} {self.activation_bytes / gib:8.3f} GiB"
            )
            lines.append(
                f"  {'total (est.)':<20} {self.total_bytes / gib:8.3f} GiB"
            )
        return "\n".join(lines)


def pytree_nbytes(tree) -> int:
    """Sum nbytes of every array leaf in a PyTree."""
    return sum(
        int(leaf.size * leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree)
    )


def measure_static_bytes(
    params: TransformerParams,
    *,
    opt_state: Any = None,
    grads: TransformerParams | None = None,
) -> MeasuredStatic:
    return MeasuredStatic(
        params_bytes=pytree_nbytes(params),
        grads_bytes=pytree_nbytes(grads) if grads is not None else None,
        optimizer_bytes=pytree_nbytes(opt_state) if opt_state is not None else None,
    )


def dtype_nbytes(dtype: jnp.dtype) -> int:
    return jnp.dtype(dtype).itemsize


def estimate_activation_bytes(
    config: GPTConfig,
    n_batch: int,
    n_seq: int,
    *,
    bytes_per_elem: int,
    use_checkpoint: bool,
) -> int:
    """Rough activation ledger: residual stream + attn scores + MLP hidden, per layer."""
    h, l, p, m = config.hidden, config.n_layers, config.n_heads, config.mlp_mult
    ff = m * h
    per_layer = (
        n_batch * n_seq * h
        + n_batch * p * n_seq * n_seq
        + n_batch * n_seq * ff
    )
    # Forward stores activations for backward; checkpointing drops most saved tensors.
    layer_factor = 2 if not use_checkpoint else 1
    return per_layer * l * bytes_per_elem * layer_factor


def estimate_memory_bytes(
    config: GPTConfig,
    n_batch: int,
    n_seq: int,
    *,
    param_dtype: jnp.dtype = jnp.float32,
    compute_dtype: jnp.dtype = jnp.float32,
    optimizer: str = "adamw",
    use_checkpoint: bool = False,
) -> MemoryBudget:
    n_params = count_params(config)
    param_b = n_params * dtype_nbytes(param_dtype)
    grad_b = n_params * dtype_nbytes(compute_dtype)
    if optimizer == "sgd":
        opt_b = 0
    elif optimizer == "adamw":
        opt_b = 2 * n_params * dtype_nbytes(jnp.float32)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    act_b = estimate_activation_bytes(
        config,
        n_batch,
        n_seq,
        bytes_per_elem=dtype_nbytes(compute_dtype),
        use_checkpoint=use_checkpoint,
    )
    return MemoryBudget(param_b, grad_b, opt_b, act_b)


def print_memory_report(
    config: GPTConfig,
    n_batch: int,
    n_seq: int,
    *,
    label: str = "",
    params: TransformerParams | None = None,
    opt_state: Any = None,
    grads: TransformerParams | None = None,
    **kwargs,
) -> MemoryBudget:
    budget = estimate_memory_bytes(config, n_batch, n_seq, **kwargs)
    measured = None
    if params is not None:
        measured = measure_static_bytes(params, opt_state=opt_state, grads=grads)
    header = f"Memory budget{' — ' + label if label else ''} (B={n_batch}, S={n_seq})"
    print(header)
    print(budget.format(measured))
    return budget


# ---------------------------------------------------------------------------
# §3 AdamW (FP32 master weights + moments)
# Ref: https://arxiv.org/abs/1711.05101 (decoupled weight decay)
# ---------------------------------------------------------------------------


class AdamWState(NamedTuple):
    first_moment: TransformerParams
    second_moment: TransformerParams


class AdamW(NamedTuple):
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    def init(self, params: TransformerParams) -> AdamWState:
        zeros = jax.tree.map(jnp.zeros_like, params)
        return AdamWState(first_moment=zeros, second_moment=zeros)

    def __call__(
        self,
        params: TransformerParams,
        grads: TransformerParams,
        state: AdamWState,
        step: Array,
    ) -> tuple[TransformerParams, AdamWState]:
        step_count = jnp.asarray(step, dtype=jnp.float32)
        first_moment = jax.tree.map(
            lambda moment, grad: self.beta1 * moment + (1.0 - self.beta1) * grad,
            state.first_moment,
            grads,
        )
        second_moment = jax.tree.map(
            lambda moment, grad: self.beta2 * moment + (1.0 - self.beta2) * jnp.square(grad),
            state.second_moment,
            grads,
        )
        first_hat = jax.tree.map(
            lambda moment: moment / (1.0 - self.beta1**step_count),
            first_moment,
        )
        second_hat = jax.tree.map(
            lambda moment: moment / (1.0 - self.beta2**step_count),
            second_moment,
        )
        params = jax.tree.map(
            lambda param, first, second: param
            * (1.0 - self.learning_rate * self.weight_decay)
            - self.learning_rate * first / (jnp.sqrt(second) + self.eps),
            params,
            first_hat,
            second_hat,
        )
        return params, AdamWState(first_moment=first_moment, second_moment=second_moment)


def sgd_update(
    params: TransformerParams,
    grads: TransformerParams,
    learning_rate: float,
) -> TransformerParams:
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)


# ---------------------------------------------------------------------------
# §4–§5 Training step (checkpoint, bf16, donate, grad accum)
# Ref checkpoint: https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html
# Ref donate:     https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
# Ref bf16:       https://cloud.google.com/tpu/docs/bfloat16
# Ref accum:      standard micro-batch trick (see script.md)
# ---------------------------------------------------------------------------

COMPUTE_DTYPE = jnp.bfloat16
MASTER_DTYPE = jnp.float32
ACCUM_STEPS = 4


def make_train_step(
    config: GPTConfig,
    optimizer: AdamW | None,
    *,
    use_checkpoint: bool,
    use_bf16: bool,
    use_donate: bool,
    accum_steps: int = 1,
):
    compute_dtype = COMPUTE_DTYPE if use_bf16 else MASTER_DTYPE
    opt = optimizer or AdamW()

    def single_grad(params, tokens_bs, targets_bs):
        def loss_fn(p):
            return loss(
                p,
                tokens_bs,
                targets_bs,
                config,
                compute_dtype=compute_dtype,
                use_checkpoint=use_checkpoint,
            )

        return jax.value_and_grad(loss_fn)(params)

    if accum_steps == 1:

        def train_step(
            params: TransformerParams,
            opt_state: AdamWState,
            tokens_bs: Array,
            targets_bs: Array,
            step: Array,
        ) -> tuple[TransformerParams, AdamWState, Array]:
            loss_val, grads = single_grad(params, tokens_bs, targets_bs)
            params, opt_state = opt(params, grads, opt_state, step)
            return params, opt_state, loss_val

    else:

        def train_step(
            params: TransformerParams,
            opt_state: AdamWState,
            tokens_bs: Array,
            targets_bs: Array,
            step: Array,
        ) -> tuple[TransformerParams, AdamWState, Array]:
            # tokens_bs shape (accum_steps, B, S) when accum_steps > 1
            def body(carry, batch):
                p, acc_g, total = carry
                tok, tgt = batch
                lv, g = single_grad(p, tok, tgt)
                acc_g = jax.tree.map(jnp.add, acc_g, g)
                return (p, acc_g, total + lv), None

            zero_grads = jax.tree.map(jnp.zeros_like, params)
            init = (params, zero_grads, jnp.array(0.0, dtype=jnp.float32))
            (params, acc_grads, total_loss), _ = jax.lax.scan(
                body, init, (tokens_bs, targets_bs)
            )
            acc_grads = jax.tree.map(lambda g: g / accum_steps, acc_grads)
            params, opt_state = opt(params, acc_grads, opt_state, step)
            return params, opt_state, total_loss / accum_steps

    if use_donate:
        # Donate only master weights — donating opt_state too can alias pytree buffers.
        return jax.jit(train_step, donate_argnums=(0,))
    return jax.jit(train_step)


# ---------------------------------------------------------------------------
# Data + demo helpers
# ---------------------------------------------------------------------------


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_corpus(data_dir: Path | None = None) -> tuple[Array, tiktoken.Encoding]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    corpus_path = data_dir / "tiny_shakespeare.txt"
    if not corpus_path.exists():
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text(
            urlopen(TINY_SHAKESPEARE_URL).read().decode(), encoding="utf-8"
        )
    text = corpus_path.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("gpt2")
    return jnp.array(enc.encode(text), dtype=jnp.int32), enc


def sample_batch(
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


def sample_micro_batches(
    key: Array,
    corpus_n: Array,
    n_batch: int,
    n_seq: int,
    accum_steps: int,
) -> tuple[Array, Array]:
    keys = jr.split(key, accum_steps)
    tokens_list = []
    targets_list = []
    for k in keys:
        tok, tgt = sample_batch(k, corpus_n, n_batch, n_seq)
        tokens_list.append(tok)
        targets_list.append(tgt)
    return jnp.stack(tokens_list), jnp.stack(targets_list)


def try_train_step(
    step_fn,
    params,
    opt_state,
    tokens_bs,
    targets_bs,
    step: int,
) -> bool:
    """Return True if step succeeds, False on OOM."""
    try:
        step_fn(
            params,
            opt_state,
            tokens_bs,
            targets_bs,
            jnp.array(step, dtype=jnp.int32),
        )
        return True
    except jax.errors.JaxRuntimeError as exc:
        if "out of memory" in str(exc).lower() or "OOM" in str(exc):
            return False
        raise


def find_max_batch(
    config: GPTConfig,
    corpus_n: Array,
    n_seq: int,
    *,
    use_checkpoint: bool,
    use_bf16: bool,
    accum_steps: int,
) -> int:
    key = jr.key(0)
    params = init_params(key, config)
    params = jax.tree.map(lambda x: x.astype(MASTER_DTYPE), params)
    opt = AdamW()
    opt_state = opt.init(params)
    step_fn = make_train_step(
        config,
        opt,
        use_checkpoint=use_checkpoint,
        use_bf16=use_bf16,
        use_donate=True,
        accum_steps=accum_steps,
    )

    lo, hi = 1, 128
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        key, key_batch = jr.split(key)
        if accum_steps == 1:
            tokens, targets = sample_batch(key_batch, corpus_n, mid, n_seq)
        else:
            tokens, targets = sample_micro_batches(
                key_batch, corpus_n, mid, n_seq, accum_steps
            )
        if try_train_step(step_fn, params, opt_state, tokens, targets, 1):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def profile_one_step(
    step_fn,
    params,
    opt_state,
    tokens_bs,
    targets_bs,
    logdir: Path,
) -> None:
    """Capture an XProf trace — open TensorBoard → Profile → Memory."""
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"XProf trace → {logdir}")
    print("  tensorboard --logdir", logdir)
    print("  Then: Profile tab → Memory (peak allocation per op)")
    jax.profiler.start_trace(str(logdir))
    step_fn(
        params,
        opt_state,
        tokens_bs,
        targets_bs,
        jnp.array(1, dtype=jnp.int32),
    )
    jax.profiler.stop_trace()


def run_demo(
    *,
    config: GPTConfig,
    n_batch: int,
    n_seq: int,
    n_steps: int,
    use_checkpoint: bool,
    use_bf16: bool,
    use_donate: bool,
    accum_steps: int,
    label: str,
) -> None:
    corpus_n, enc = load_corpus()
    print(f"corpus: {corpus_n.shape[0]:,} tokens · vocab {enc.n_vocab:,}")

    key = jr.key(0)
    params = init_params(jr.key(1), config)
    params = jax.tree.map(lambda x: x.astype(MASTER_DTYPE), params)
    opt = AdamW()
    opt_state = opt.init(params)

    # One backward pass so we can measure grad bytes from the real PyTree.
    key, key_batch = jr.split(key)
    tokens_bs, targets_bs = sample_batch(key_batch, corpus_n, n_batch, n_seq)
    compute_dtype = COMPUTE_DTYPE if use_bf16 else MASTER_DTYPE

    def loss_fn(p: TransformerParams) -> Array:
        return loss(
            p,
            tokens_bs,
            targets_bs,
            config,
            compute_dtype=compute_dtype,
            use_checkpoint=use_checkpoint,
        )

    _, grads = jax.value_and_grad(loss_fn)(params)

    print_memory_report(
        config,
        n_batch,
        n_seq,
        param_dtype=MASTER_DTYPE,
        compute_dtype=compute_dtype,
        optimizer="adamw",
        use_checkpoint=use_checkpoint,
        label=label,
        params=params,
        opt_state=opt_state,
        grads=grads,
    )

    step_fn = make_train_step(
        config,
        opt,
        use_checkpoint=use_checkpoint,
        use_bf16=use_bf16,
        use_donate=use_donate,
        accum_steps=accum_steps,
    )

    eff_batch = n_batch * accum_steps
    print(
        f"\nTraining {n_steps} steps — {label}\n"
        f"  micro_batch={n_batch}  accum={accum_steps}  effective_batch={eff_batch}  "
        f"seq={n_seq}  checkpoint={use_checkpoint}  bf16={use_bf16}  donate={use_donate}"
    )

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        key, key_batch = jr.split(key)
        if accum_steps == 1:
            tokens, targets = sample_batch(key_batch, corpus_n, n_batch, n_seq)
        else:
            tokens, targets = sample_micro_batches(
                key_batch, corpus_n, n_batch, n_seq, accum_steps
            )
        params, opt_state, loss_val = step_fn(
            params,
            opt_state,
            tokens,
            targets,
            jnp.array(step, dtype=jnp.int32),
        )
        if step == 1 or step % max(1, n_steps // 5) == 0 or step == n_steps:
            print(f"  step {step:3d}  loss {float(loss_val):.4f}")
    elapsed = time.perf_counter() - t0
    print(f"  wall time {elapsed:.1f}s ({elapsed / n_steps * 1000:.0f} ms/step)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode 8 — memory & mixed precision")
    parser.add_argument("--profile", action="store_true", help="run one XProf trace")
    parser.add_argument(
        "--find-max-batch",
        action="store_true",
        help="binary-search max micro-batch before OOM",
    )
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    corpus_n, enc = load_corpus()
    config = stress_config(enc.n_vocab)
    n_seq = config.max_ctx

    if args.find_max_batch:
        print("=== Baseline (fp32, no checkpoint) ===")
        b0 = find_max_batch(
            config, corpus_n, n_seq, use_checkpoint=False, use_bf16=False, accum_steps=1
        )
        print(f"max batch: {b0}\n")

        print("=== + activation checkpoint ===")
        b1 = find_max_batch(
            config, corpus_n, n_seq, use_checkpoint=True, use_bf16=False, accum_steps=1
        )
        print(f"max batch: {b1}\n")

        print("=== + checkpoint + bf16 ===")
        b2 = find_max_batch(
            config, corpus_n, n_seq, use_checkpoint=True, use_bf16=True, accum_steps=1
        )
        print(f"max batch: {b2}\n")

        print("=== + checkpoint + bf16 + grad accum (×4) ===")
        b3 = find_max_batch(
            config,
            corpus_n,
            n_seq,
            use_checkpoint=True,
            use_bf16=True,
            accum_steps=ACCUM_STEPS,
        )
        print(f"max micro-batch (effective ×{ACCUM_STEPS}): {b3}")
        return

    if args.profile:
        n_batch = 8
        key = jr.key(0)
        params = init_params(key, config)
        params = jax.tree.map(lambda x: x.astype(MASTER_DTYPE), params)
        opt = AdamW()
        opt_state = opt.init(params)
        step_fn = make_train_step(
            config,
            opt,
            use_checkpoint=True,
            use_bf16=True,
            use_donate=True,
            accum_steps=1,
        )
        tokens, targets = sample_batch(jr.key(1), corpus_n, n_batch, n_seq)
        profile_one_step(
            step_fn,
            params,
            opt_state,
            tokens,
            targets,
            Path("/tmp/ep09_xprof"),
        )
        return

    # Progressive demo — each stage adds one lever (compare max batch mentally or use --find-max-batch)
    run_demo(
        config=config,
        n_batch=4,
        n_seq=n_seq,
        n_steps=args.steps,
        use_checkpoint=False,
        use_bf16=False,
        use_donate=False,
        accum_steps=1,
        label="baseline fp32",
    )
    run_demo(
        config=config,
        n_batch=8,
        n_seq=n_seq,
        n_steps=args.steps,
        use_checkpoint=True,
        use_bf16=False,
        use_donate=False,
        accum_steps=1,
        label="+ activation checkpoint",
    )
    run_demo(
        config=config,
        n_batch=16,
        n_seq=n_seq,
        n_steps=args.steps,
        use_checkpoint=True,
        use_bf16=True,
        use_donate=True,
        accum_steps=1,
        label="+ bf16 + donate",
    )
    run_demo(
        config=config,
        n_batch=16,
        n_seq=n_seq,
        n_steps=args.steps,
        use_checkpoint=True,
        use_bf16=True,
        use_donate=True,
        accum_steps=ACCUM_STEPS,
        label="+ grad accumulation",
    )


if __name__ == "__main__":
    main()
