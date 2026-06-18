"""Reflecting-digits spine dataset from Episode 0.

Sequence: 012345678987654321012345678987654321...
Period 18 over digit vocabulary {0, ..., 9}.
"""

from __future__ import annotations

import jax.numpy as jnp

VOCAB_SIZE = 10
PERIOD = 18

# One period of the reflecting walk.
REFLECTING_PERIOD = jnp.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    dtype=jnp.int32,
)


def make_reflecting_digits(seq_len: int) -> jnp.ndarray:
    """Return ``seq_len`` tokens from the reflecting-digits sequence."""
    idx = jnp.arange(seq_len, dtype=jnp.int32) % PERIOD
    return REFLECTING_PERIOD[idx]


def train_val_split(
    tokens: jnp.ndarray, train_frac: float = 0.8
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split a token stream into contiguous train / validation segments."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    n_train = int(len(tokens) * train_frac)
    return tokens[:n_train], tokens[n_train:]


def decode(tokens: jnp.ndarray) -> str:
    """Render digit tokens as a string (for quick inspection)."""
    return "".join(str(int(t)) for t in tokens)
