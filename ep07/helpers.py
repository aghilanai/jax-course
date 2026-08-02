"""Episode 7 — model, data, optimizers, and train helpers (single module)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import tiktoken
from jax import Array

Params = Any
OptState = Any

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_corpus(data_dir: Path | None = None) -> tuple[Array, tiktoken.Encoding]:
    """Load (or download) Tiny Shakespeare; return token ids and GPT-2 encoder."""
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


def train_val_split(
    corpus_n: Array, val_frac: float = 0.1
) -> tuple[Array, Array]:
    """Contiguous hold-out: last `val_frac` of tokens are validation."""
    n = corpus_n.shape[0]
    n_val = int(n * val_frac)
    return corpus_n[: n - n_val], corpus_n[n - n_val :]


def sample_batch(
    key: Array, corpus_n: Array, n_batch: int, n_seq: int
) -> tuple[Array, Array]:
    """Random (B, S) windows → inputs / next-token targets."""
    n_tokens = corpus_n.shape[0]
    starts_b = jr.randint(key, (n_batch,), 0, n_tokens - n_seq - 1)
    offsets_s1 = jnp.arange(n_seq + 1)
    idx_bs1 = starts_b[:, None] + offsets_s1[None, :]
    chunks_bs1 = corpus_n[idx_bs1]
    return chunks_bs1[:, :-1], chunks_bs1[:, 1:]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GPTConfig(NamedTuple):
    """Hyperparameters for a GPT-2-style decoder."""

    vocab_dim: int = 50_257
    max_ctx: int = 256
    n_layers: int = 4
    n_heads: int = 8
    hidden: int = 256
    mlp_mult: int = 4
    dropout_rate: float = 0.2


def ep07_config(vocab_dim: int, dropout_rate: float = 0.2) -> GPTConfig:
    """~16M params (4L×256H×8 heads) for short demos."""
    return GPTConfig(
        vocab_dim=vocab_dim,
        max_ctx=256,
        n_layers=4,
        n_heads=8,
        hidden=256,
        mlp_mult=4,
        dropout_rate=dropout_rate,
    )


def dropout(
    x: Array, key: Array | None, rate: float, *, train: bool
) -> Array:
    """Inverted dropout. No-op when `train` is False or `rate` is 0."""
    if (not train) or rate == 0.0:
        return x
    if key is None:
        raise ValueError("dropout requires a PRNG key when train=True and rate>0")
    keep = 1.0 - rate
    mask = jr.bernoulli(key, keep, x.shape).astype(x.dtype)
    return mask * x / keep


class LayerNormParams(NamedTuple):
    gamma_h: Array  # (H,)
    beta_h: Array  # (H,)

    @classmethod
    def init(cls, key: Array, hidden: int) -> LayerNormParams:
        del key
        return cls(gamma_h=jnp.ones(hidden), beta_h=jnp.zeros(hidden))

    def __call__(self, x_bsh: Array) -> Array:
        mean_h = jnp.mean(x_bsh, axis=-1, keepdims=True)
        var_h = jnp.var(x_bsh, axis=-1, keepdims=True)
        x_hat_bsh = (x_bsh - mean_h) / jnp.sqrt(var_h + 1e-5)
        return self.gamma_h * x_hat_bsh + self.beta_h


class LinearParams(NamedTuple):
    w_io: Array  # (I, O)
    b_o: Array  # (O,)

    @classmethod
    def init(
        cls, key: Array, n_in: int, n_out: int, std: float = 0.02
    ) -> LinearParams:
        w_io = jr.normal(key, (n_in, n_out)) * std
        return cls(w_io=w_io, b_o=jnp.zeros(n_out))

    def __call__(self, x_bsh: Array) -> Array:
        return x_bsh @ self.w_io + self.b_o


class MultiHeadAttentionParams(NamedTuple):
    proj_q: LinearParams
    proj_k: LinearParams
    proj_v: LinearParams
    proj_o: LinearParams

    @classmethod
    def init(cls, key: Array, hidden: int, residual_scale: float = 1.0) -> MultiHeadAttentionParams:
        key_q, key_k, key_v, key_o = jr.split(key, 4)
        return cls(
            proj_q=LinearParams.init(key_q, hidden, hidden),
            proj_k=LinearParams.init(key_k, hidden, hidden),
            proj_v=LinearParams.init(key_v, hidden, hidden),
            # GPT-2 residual-path scaling: shrink output projection by 1/√N
            proj_o=LinearParams.init(key_o, hidden, hidden, std=0.02 * residual_scale),
        )

    def __call__(self, x_bsh: Array, *, n_heads: int, mask_ss: Array) -> Array:
        n_batch, n_seq, hidden = x_bsh.shape
        n_head_dim = hidden // n_heads

        q_bsh = self.proj_q(x_bsh)
        k_bsh = self.proj_k(x_bsh)
        v_bsh = self.proj_v(x_bsh)

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
        return self.proj_o(out_bsh)


def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


class MLPParams(NamedTuple):
    fc_hf: LinearParams  # (H, F)
    fc_fh: LinearParams  # (F, H)

    @classmethod
    def init(
        cls, key: Array, hidden: int, mlp_mult: int, residual_scale: float = 1.0
    ) -> MLPParams:
        n_ff = mlp_mult * hidden
        key_up, key_down = jr.split(key)
        return cls(
            fc_hf=LinearParams.init(key_up, hidden, n_ff),
            fc_fh=LinearParams.init(key_down, n_ff, hidden, std=0.02 * residual_scale),
        )

    def __call__(self, x_bsh: Array) -> Array:
        return self.fc_fh(gelu(self.fc_hf(x_bsh)))


class BlockParams(NamedTuple):
    ln1_h: LayerNormParams
    attn_hh: MultiHeadAttentionParams
    ln2_h: LayerNormParams
    mlp_hh: MLPParams

    @classmethod
    def init(
        cls, key: Array, hidden: int, mlp_mult: int, residual_scale: float = 1.0
    ) -> BlockParams:
        key_ln1, key_attn, key_ln2, key_mlp = jr.split(key, 4)
        return cls(
            ln1_h=LayerNormParams.init(key_ln1, hidden),
            attn_hh=MultiHeadAttentionParams.init(
                key_attn, hidden, residual_scale=residual_scale
            ),
            ln2_h=LayerNormParams.init(key_ln2, hidden),
            mlp_hh=MLPParams.init(
                key_mlp, hidden, mlp_mult, residual_scale=residual_scale
            ),
        )

    def __call__(
        self,
        x_bsh: Array,
        *,
        n_heads: int,
        mask_ss: Array,
        key: Array | None,
        dropout_rate: float,
        train: bool,
    ) -> Array:
        key_attn = key_mlp = None
        if train and dropout_rate > 0.0:
            key_attn, key_mlp = jr.split(key)

        attn_bsh = self.attn_hh(self.ln1_h(x_bsh), n_heads=n_heads, mask_ss=mask_ss)
        x_bsh = x_bsh + dropout(attn_bsh, key_attn, dropout_rate, train=train)
        mlp_bsh = self.mlp_hh(self.ln2_h(x_bsh))
        x_bsh = x_bsh + dropout(mlp_bsh, key_mlp, dropout_rate, train=train)
        return x_bsh


def causal_mask(n_seq: int) -> Array:
    return jnp.tril(jnp.ones((n_seq, n_seq)))


class TransformerParams(NamedTuple):
    embed_vh: Array  # (V, H)
    pos_embed_sh: Array  # (S, H)
    blocks_l: tuple[BlockParams, ...]
    ln_final_h: LayerNormParams

    @classmethod
    def init(cls, key: Array, config: GPTConfig) -> TransformerParams:
        key_embed, key_pos, key_blocks = jr.split(key, 3)
        # N = 2 * n_layers residual branches (attn + MLP per block)
        residual_scale = (2.0 * config.n_layers) ** -0.5
        block_keys = jr.split(key_blocks, config.n_layers)
        blocks_l = tuple(
            BlockParams.init(k, config.hidden, config.mlp_mult, residual_scale)
            for k in block_keys
        )
        return cls(
            embed_vh=jr.normal(key_embed, (config.vocab_dim, config.hidden)) * 0.02,
            pos_embed_sh=jr.normal(key_pos, (config.max_ctx, config.hidden)) * 0.01,
            blocks_l=blocks_l,
            ln_final_h=LayerNormParams.init(key_blocks, config.hidden),
        )

    def __call__(
        self,
        tokens_bs: Array,
        *,
        config: GPTConfig,
        key: Array | None = None,
        train: bool = False,
    ) -> Array:
        """Return logits_bsv with shape (B, S, V).

        Pass `train=True` and a PRNG `key` to enable dropout; eval leaves both off.
        """
        n_batch, n_seq = tokens_bs.shape
        if n_seq > config.max_ctx:
            raise ValueError(
                f"sequence length {n_seq} exceeds max_ctx {config.max_ctx}"
            )

        pos_s = jnp.arange(n_seq)
        x_bsh = self.embed_vh[tokens_bs] + self.pos_embed_sh[pos_s]
        mask_ss = causal_mask(n_seq)

        use_dropout = train and config.dropout_rate > 0.0
        if use_dropout:
            key, key_emb = jr.split(key)
            x_bsh = dropout(x_bsh, key_emb, config.dropout_rate, train=True)

        for block in self.blocks_l:
            key_block = None
            if use_dropout:
                key, key_block = jr.split(key)
            x_bsh = block(
                x_bsh,
                n_heads=config.n_heads,
                mask_ss=mask_ss,
                key=key_block,
                dropout_rate=config.dropout_rate,
                train=train,
            )

        x_bsh = self.ln_final_h(x_bsh)
        return x_bsh @ self.embed_vh.T  # weight-tied LM head


def cross_entropy_loss(logits_bsv: Array, targets_bs: Array) -> Array:
    log_probs_bsv = jax.nn.log_softmax(logits_bsv, axis=-1)
    return -jnp.mean(
        jnp.take_along_axis(log_probs_bsv, targets_bs[..., None], axis=-1)
    )


def loss(
    params: TransformerParams,
    tokens_bs: Array,
    targets_bs: Array,
    config: GPTConfig,
    *,
    key: Array | None = None,
    train: bool = False,
) -> Array:
    return cross_entropy_loss(
        params(tokens_bs, config=config, key=key, train=train),
        targets_bs,
    )


def count_params(params: TransformerParams) -> int:
    return int(sum(leaf.size for leaf in jax.tree.leaves(params)))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Optimizer helpers (implement the optimizers themselves in the notebooks)
# ---------------------------------------------------------------------------


def map_leaves(update_fn, params: Params, grads: Params, *state_trees: Params):
    """Apply `update_fn` to each param leaf; state trees must match `params` structure."""
    param_leaves, treedef = jax.tree_util.tree_flatten(params)
    grad_leaves = treedef.flatten_up_to(grads)
    state_leaf_lists = [treedef.flatten_up_to(tree) for tree in state_trees]
    new_param_leaves = []
    new_state_leaf_lists: list[list[Array]] = [[] for _ in state_trees]
    for i, (param, grad) in enumerate(zip(param_leaves, grad_leaves)):
        outs = update_fn(param, grad, *[leaves[i] for leaves in state_leaf_lists])
        new_param_leaves.append(outs[0])
        for j, state_leaf in enumerate(outs[1:]):
            new_state_leaf_lists[j].append(state_leaf)
    new_params = treedef.unflatten(new_param_leaves)
    new_states = tuple(treedef.unflatten(leaves) for leaves in new_state_leaf_lists)
    return new_params, *new_states


def newton_schulz5(matrix: Array, steps: int = 5) -> Array:
    """Approximate orthogonalization of a 2D matrix (Muon / zeroth-power).

    Quintic Newton–Schulz coefficients from Keller Jordan's Muon
    (https://kellerjordan.github.io/posts/muon/).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.astype(jnp.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.T
        x = a * x + (b * a_mat + c * (a_mat @ a_mat)) @ x
    if transposed:
        x = x.T
    return x.astype(matrix.dtype)


# Duck-typed: any NamedTuple with `.name`, `.learning_rate`, `.init`, `__call__`.
Optimizer = Any


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


class TrainHistory(NamedTuple):
    name: str
    steps: list[int]
    train_loss: list[float]
    val_loss: list[float]
    train_ppl: list[float]
    val_ppl: list[float]


def make_train_step(optimizer: Optimizer, config: GPTConfig):
    """Build a jitted step: (params, opt_state, tokens, targets, step, key) → …"""

    @jax.jit
    def train_step(
        params: TransformerParams,
        opt_state,
        tokens_bs: Array,
        targets_bs: Array,
        step: Array,
        key: Array,
    ):
        def loss_fn(params_in: TransformerParams) -> Array:
            return loss(
                params_in,
                tokens_bs,
                targets_bs,
                config,
                key=key,
                train=True,
            )

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        # Skip update on non-finite loss (stability guard)
        finite = jnp.isfinite(loss_val)

        def do_update(args):
            params_in, state_in, grads_in, step_in = args
            return optimizer(params_in, grads_in, state_in, step_in)

        def skip_update(args):
            params_in, state_in, _grads_in, _step_in = args
            return params_in, state_in

        params, opt_state = jax.lax.cond(
            finite,
            do_update,
            skip_update,
            (params, opt_state, grads, step),
        )
        return params, opt_state, loss_val

    return train_step


def eval_loss(
    params: TransformerParams,
    corpus_n: Array,
    config: GPTConfig,
    key: Array,
    n_batch: int,
    n_seq: int,
    n_batches: int = 4,
) -> Array:
    """Mean loss over a few random val batches (dropout off)."""
    losses = []
    for i in range(n_batches):
        key, sub = jr.split(key)
        tokens_bs, targets_bs = sample_batch(sub, corpus_n, n_batch, n_seq)
        losses.append(loss(params, tokens_bs, targets_bs, config, train=False))
    return jnp.mean(jnp.stack(losses))


def train(
    *,
    optimizer: Optimizer,
    params: TransformerParams,
    config: GPTConfig,
    train_corpus: Array,
    val_corpus: Array,
    key: Array,
    n_steps: int = 200,
    n_batch: int = 64,
    n_seq: int = 128,
    log_every: int = 100,
    eval_batches: int = 4,
) -> tuple[TransformerParams, TrainHistory]:
    """Run `n_steps` of training with the given optimizer; return params + curves."""
    opt_state = optimizer.init(params)
    step_fn = make_train_step(optimizer, config)

    hist = TrainHistory(
        name=optimizer.name,
        steps=[],
        train_loss=[],
        val_loss=[],
        train_ppl=[],
        val_ppl=[],
    )

    tokens_per_step = n_batch * n_seq
    print(
        f"[{optimizer.name}] training {n_steps} steps "
        f"(batch={n_batch}, seq={n_seq}, ~{tokens_per_step * n_steps / 1e6:.1f}M tokens, "
        f"lr={optimizer.learning_rate}, dropout={config.dropout_rate}) …",
        flush=True,
    )

    running_train = 0.0
    running_count = 0

    for step_i in range(1, n_steps + 1):
        key, key_batch, key_drop = jr.split(key, 3)
        tokens_bs, targets_bs = sample_batch(
            key_batch, train_corpus, n_batch, n_seq
        )
        params, opt_state, loss_val = step_fn(
            params,
            opt_state,
            tokens_bs,
            targets_bs,
            jnp.array(step_i, dtype=jnp.int32),
            key_drop,
        )
        loss_f = float(loss_val)
        running_train += loss_f
        running_count += 1

        if step_i == 1 or step_i % log_every == 0 or step_i == n_steps:
            key, key_val = jr.split(key)
            avg_train = running_train / max(running_count, 1)
            val_l = float(
                eval_loss(
                    params,
                    val_corpus,
                    config,
                    key_val,
                    n_batch,
                    n_seq,
                    n_batches=eval_batches,
                )
            )
            hist.steps.append(step_i)
            hist.train_loss.append(avg_train)
            hist.val_loss.append(val_l)
            hist.train_ppl.append(float(jnp.exp(avg_train)))
            hist.val_ppl.append(float(jnp.exp(val_l)))
            print(
                f"  step {step_i:4d}  train_loss {avg_train:.4f}  "
                f"val_loss {val_l:.4f}  val_ppl {hist.val_ppl[-1]:.1f}",
                flush=True,
            )
            running_train = 0.0
            running_count = 0

    return params, hist


def plot_histories(
    histories: list[TrainHistory],
    out_path: Path | None = None,
    title: str = "Episode 7 — Optimizer comparison (Tiny Shakespeare)",
) -> Path:
    """Plot train/val loss for each optimizer; save PNG."""
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "optimizer_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=120)

    for hist in histories:
        axes[0].plot(hist.steps, hist.train_loss, label=hist.name, linewidth=2)
        axes[1].plot(hist.steps, hist.val_loss, label=hist.name, linewidth=2)

    axes[0].set_title("Train loss")
    axes[1].set_title("Val loss")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path
