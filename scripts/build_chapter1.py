#!/usr/bin/env python3
"""Build Part I (Chapter 1) solution notebooks ep01–ep05."""

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


def write_ep(name: str, cells: list) -> None:
    ep = ROOT / name
    ep.mkdir(parents=True, exist_ok=True)
    path = ep / "solution.ipynb"
    path.write_text(json.dumps(nb(cells), indent=1) + "\n")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_student.py"), name],
        check=True,
    )


EP01 = [
    md(
        """# Episode 1 — JAX as a Functional Array Accelerator

**Instructor notebook** · run top-to-bottom before recording.

Why JAX is not NumPy, and why that matters for ML.

| | |
|---|---|
| **Chapter** | 1.1 · Part I — Pure JAX |
| **Prereq** | Python, basic NumPy |
| **Next** | Episode 2 — JIT, tracing, and the jaxpr |

**JAX docs:** [JAX 101](https://jax.readthedocs.io/en/latest/jax-101.html) · [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html) · [Pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html) · [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) · [Async dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html)
"""
    ),
    code("import jax\nimport jax.numpy as jnp\nimport jax.random as jr\nimport numpy as np"),
    md(
        """## Pure functions and immutability

JAX has no hidden global state. Arrays are **immutable** — updates return new arrays via `.at[...].set(...)`. This purity is what makes `jit`, `grad`, and `vmap` composable later."""
    ),
    code(
        """x = jnp.array([1.0, 2.0, 3.0])

try:
    x[0] = 99.0
except TypeError as e:
    print(type(e).__name__, ":", e)

x_new = x.at[0].set(99.0)
print("original:", x)
print("updated: ", x_new)"""
    ),
    md(
        """## PRNG keys

JAX has **no hidden global RNG**. Create a **key**, **`split`** before every random draw, and never reuse a consumed subkey."""
    ),
    code(
        """key = jr.key(0)
key, subkey = jr.split(key)

a = jr.normal(subkey, (3,))
print(a)

# Same root key → same draws
key = jr.key(0)
key, subkey = jr.split(key)
b = jr.normal(subkey, (3,))
print(jnp.allclose(a, b))"""
    ),
    md(
        """## `jnp` vs NumPy — same API, different execution

NumPy runs eagerly on the host. JAX builds a deferred computation graph (we compile it in Episode 2). Start with the same matmul in both."""
    ),
    code(
        """M, K, N = 1024, 512, 1024

a_np = np.random.randn(M, K).astype(np.float32)
b_np = np.random.randn(K, N).astype(np.float32)

# NumPy — eager on CPU
c_np = a_np @ b_np
print("NumPy matmul shape:", c_np.shape)

%timeit a_np @ b_np"""
    ),
    code(
        """key = jax.random.key(0)
key, k_a, k_b = jax.random.split(key, 3)
a_jnp = jax.random.normal(k_a, (M, K))
b_jnp = jax.random.normal(k_b, (K, N))

# JAX — dispatches asynchronously; block to measure
c_jnp = a_jnp @ b_jnp
jax.block_until_ready(c_jnp)
print("JAX matmul shape:", c_jnp.shape)
print("match (same problem size):", np.allclose(np.array(c_jnp), np.array(a_jnp @ b_jnp), atol=1e-4))

%timeit jax.block_until_ready(a_jnp @ b_jnp)"""
    ),
    md(
        """## Device placement

Arrays live on a **device** (CPU, GPU, TPU). Use `jax.devices()` to see what's available; `jax.device_put` moves data explicitly."""
    ),
    code(
        """print("devices:", jax.devices())
print("default backend:", jax.default_backend())

x_host = jnp.ones((4, 4))
x_device = jax.device_put(x_host, jax.devices()[0])
print("array device:", x_device.devices())"""
    ),
    md(
        """## Asynchronous dispatch

JAX returns before work finishes on accelerator hardware. Always call `jax.block_until_ready(...)` before stopping a timer."""
    ),
    code(
        """a = jax.random.normal(jax.random.key(1), (2048, 2048))
b = jax.random.normal(jax.random.key(2), (2048, 2048))

print("without block_until_ready (may under-report on GPU):")
%timeit a @ b

print("with block_until_ready:")
%timeit jax.block_until_ready(a @ b)"""
    ),
    md(
        """## Broadcasting

JAX follows [NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html): compare shapes from the **right**; dimensions match if equal or one is `1`.

For a batched matmul `x @ w` with `x` shape `(B, D_in)`, `w` shape `(D_in, D_out)`, the result is `(B, D_out)`. A bias `b` with shape `(D_out,)` broadcasts across the batch rows — no extra storage."""
    ),
    code(
        """B, d_in, d_out = 64, 10, 10
key = jr.key(5)
key, k_x = jr.split(key)
x = jr.normal(k_x, (B, d_in))
w = jr.normal(jr.key(6), (d_in, d_out)) * 0.1
b = jnp.zeros(d_out)

logits = x @ w
print("broadcast_shapes:", jnp.broadcast_shapes(logits.shape, b.shape))
y = logits + b
print("y.shape:", y.shape)
print("matches row-wise add:", jnp.allclose(y, logits + b[None, :]))"""
    ),
    md(
        """## Deferred execution preview — shape-sensitive functions

JAX traces functions on **shapes and dtypes**. Calling the same Python function with different array shapes can re-trace (Episode 2 covers this in depth). Here, a side-effect counter shows retracing."""
    ),
    code(
        """trace_count = 0


def shape_logger(x):
    global trace_count
    trace_count += 1
    print(f"  Python body ran (trace #{trace_count}), x.shape = {x.shape}")
    return jnp.sum(x ** 2)


x_small = jnp.ones((8,))
x_large = jnp.ones((16,))

print("call 1:")
y1 = shape_logger(x_small)
print("call 2 (same shape):")
y2 = shape_logger(x_small)
print("call 3 (new shape):")
y3 = shape_logger(x_large)
print(f"Python body ran {trace_count} times — new shapes re-run Python (until we jit in Ep 2)")"""
    ),
    md(
        """> **Key insight:** The purity constraint is not a limitation — it is what makes compilation, differentiation, and parallelism composable. Every later chapter depends on this."""
    ),
    md(
        """---
## Exercise

1. Time NumPy vs JAX matmul on your machine (adjust `M, K, N` if needed).
2. `split` a key and draw twice from the same root — verify reproducibility.
3. Print `jax.devices()` and the device of one `device_put` array.
4. Show `(B, D_out) + (D_out,)` broadcasting with `jnp.broadcast_shapes`.
5. Benchmark one large matmul with and without `block_until_ready`.

*(Solution below.)*"""
    ),
    code(
        """# Demo answers — students replicate the cells above with their own timings.
print("devices:", [str(d) for d in jax.devices()])
print("backend:", jax.default_backend())"""
    ),
    md("**Next:** Episode 2 — JIT, tracing, and the jaxpr."),
]

EP02 = [
    md(
        """# Episode 2 — JIT, Tracing, and the Jaxpr

**Instructor notebook** · run top-to-bottom before recording.

`jax.jit` traces your function once into a **jaxpr**, lowers to XLA, and caches the executable.

| | |
|---|---|
| **Chapter** | 1.2 · Part I — Pure JAX |
| **Prereq** | Episode 1 |
| **Next** | Episode 3 — automatic differentiation |

**JAX docs:** [Tracing](https://docs.jax.dev/en/latest/tracing.html) · [jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) · [`jax.make_jaxpr`](https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html) · [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)
"""
    ),
    code(
        """import time

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import make_jaxpr"""
    ),
    md(
        """## What `jit` does

On the first call with given **shapes/dtypes**, JAX **traces** your function: values become tracers, ops record into a **jaxpr**. That jaxpr lowers to **StableHLO**, XLA optimizes it, and later calls reuse the cached executable. New shapes → **retrace**."""
    ),
    md("## A 3-layer MLP to inspect"),
    code(
        """def mlp3(params, x):
    for layer in params:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x


def init_mlp(key, d_in, d_hidden, d_out):
    key, k0, k1, k2 = jr.split(key, 4)
    return [
        {"w": jr.normal(k0, (d_in, d_hidden)) * 0.1, "b": jnp.zeros(d_hidden)},
        {"w": jr.normal(k1, (d_hidden, d_hidden)) * 0.1, "b": jnp.zeros(d_hidden)},
        {"w": jr.normal(k2, (d_hidden, d_out)) * 0.1, "b": jnp.zeros(d_out)},
    ]


key = jr.key(0)
params = init_mlp(key, 16, 32, 4)
x = jr.normal(jr.key(1), (8, 16))

print(make_jaxpr(mlp3)(params, x))"""
    ),
    md("## `jit` wraps the jaxpr in a compiled call"),
    code(
        """jitted_mlp = jax.jit(mlp3)
print(make_jaxpr(jitted_mlp)(params, x))"""
    ),
    md("## First call vs steady state"),
    code(
        """def time_call(fn, *args):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) * 1000, out


fresh = jax.jit(mlp3)
first_ms, _ = time_call(fresh, params, x)
for _ in range(3):
    fresh(params, x)
steady_ms, _ = time_call(fresh, params, x)
print(f"first call (compile+run): {first_ms:.2f} ms")
print(f"steady state (run only):  {steady_ms:.2f} ms")"""
    ),
    md("## Retrace on shape change"),
    code(
        """compile_log = []


def mlp_trace(params, x):
    compile_log.append(x.shape)
    return mlp3(params, x)


jitted_trace = jax.jit(mlp_trace)
x_a = jr.normal(jr.key(2), (4, 16))
x_b = jr.normal(jr.key(3), (8, 16))

jitted_trace(params, x_a)
jitted_trace(params, x_a)
jitted_trace(params, x_b)
print("shapes seen at trace time:", compile_log)"""
    ),
    md(
        """## Static vs traced — what `jit` can and cannot specialize on

| Kind | Examples | At `jit` time |
|------|----------|----------------|
| **Traced** | `jnp` arrays | Shapes and dtypes are baked into the compiled program. New shapes → **retrace**. |
| **Static** | Python `int`, `bool`, `str`, `None` that control structure | Must mark with `static_argnums` / `static_argnames` or they become tracers and break Python control flow. |
| **Not compiled** | `print`, file I/O, mutation, arbitrary Python loops over Python lists | Run at **trace** time only (or not at all inside compiled code). |

**Rule of thumb:** tensor data → traced; hyperparameters and flags that pick branches → static."""
    ),
    md("## `static_argnums` for Python scalars"),
    code(
        """def repeat_matmul(x, n):
    for _ in range(n):
        x = x @ x
    return x


# n is a Python int — mark static so changing n does not confuse the batch dim
jitted_repeat = jax.jit(repeat_matmul, static_argnums=1)

a = jr.normal(jr.key(4), (64, 64))
print("n=2:", jitted_repeat(a, 2).shape)
print("n=3:", jitted_repeat(a, 3).shape)"""
    ),
    md("## Print fires at **trace** time, not every run"),
    code(
        """print("--- outside jit ---")


@jax.jit
def noisy_square(x):
    print("inside jitted function, x.shape =", x.shape)
    return x ** 2


v = jnp.ones((3,))
print("run 1:")
noisy_square(v)
print("run 2 (same shape):")
noisy_square(v)
print("run 3 (new shape):")
noisy_square(jnp.ones((5,)))"""
    ),
    md(
        """> **Key insight:** JIT does not execute your Python code at runtime — it traces it once. Print statements inside jitted functions fire at trace time, not run time."""
    ),
    md(
        """---
## Exercise

1. `make_jaxpr` on the 3-layer MLP — find a `dot_general` or `dot` primitive.
2. Change batch size and log how many distinct shapes your jitted function sees.
3. Pass a Python `int` loop count with `static_argnums`.
4. Time first vs second `jit` call on the same shapes.
5. List one traced argument and one static argument from your MLP example.

*(Solution below.)*"""
    ),
    code(
        """print("jaxpr snippet:")
print(str(make_jaxpr(jitted_mlp)(params, x))[:400], "...")"""
    ),
    md("**Next:** Episode 3 — `grad`, `value_and_grad`, and checkpointing."),
]

EP03 = [
    md(
        """# Episode 3 — Automatic Differentiation

**Instructor notebook** · run top-to-bottom before recording.

`jax.grad` for scalars; `value_and_grad` for training; `jax.checkpoint` to trade compute for memory.

| | |
|---|---|
| **Chapter** | 1.3 · Part I — Pure JAX |
| **Prereq** | Episodes 1–2 |
| **Next** | Episode 4 — `vmap`, `scan`, and vectorization |

**JAX docs:** [Automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) · [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html) · [`jax.vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html) · [`jax.value_and_grad`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html) · [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html)
"""
    ),
    code(
        """import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, value_and_grad"""
    ),
    md(
        """## `grad` — scalar outputs only

`jax.grad(f)` returns a function that computes ∂f/∂x. The output of `f` must be a **scalar** (a 0-D array)."""
    ),
    code(
        """def loss_wrt_W(W, x):
    probs = jax.nn.softmax(x)
    return jnp.sum(probs @ W)


x = jnp.array([1.0, 2.0, 0.5])
W = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

grad_W = grad(loss_wrt_W, argnums=0)(W, x)
print("grad W shape:", grad_W.shape)
print(grad_W)"""
    ),
    md("## `value_and_grad` — loss and gradients together"),
    code(
        """def ce_loss(params, x, y):
    logits = x @ params
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[y]


params = jr.normal(jr.key(0), (4, 3))
x = jr.normal(jr.key(1), (4,))
y = jnp.int32(1)

loss, grads = value_and_grad(ce_loss)(params, x, y)
print("loss:", loss)
print("grads shape:", grads.shape)"""
    ),
    md("## Check against finite differences"),
    code(
        """def finite_diff_grad(f, x, eps=1e-4):
    g = jnp.zeros_like(x)
    for idx in jnp.ndindex(x.shape):
        xp = x.at[idx].add(eps)
        xm = x.at[idx].add(-eps)
        g = g.at[idx].set((f(xp) - f(xm)) / (2 * eps))
    return g


_, ad_grad = value_and_grad(ce_loss)(params, x, y)
fd_grad = finite_diff_grad(lambda p: ce_loss(p, x, y), params)
print("max |AD - FD|:", float(jnp.max(jnp.abs(ad_grad - fd_grad))))"""
    ),
    md(
        """## `vjp` when the output is not a scalar

`jax.grad` requires a **scalar** loss. For vector- or tuple-valued functions, use `jax.vjp` — it returns the output plus a function that maps a **cotangent** (same shape as the output) back to input gradients."""
    ),
    code(
        """def vec_out(x):
    return jnp.array([jnp.sum(x ** 2), jnp.sum(x)])


x = jnp.array([1.0, 2.0, 3.0])

try:
    grad(vec_out)(x)
except TypeError as e:
    print("grad on vector output:", e)

out, vjp_fn = jax.vjp(vec_out, x)
# Cotangent selects which output component to backprop — here weight both equally
grad_x = vjp_fn(jnp.array([1.0, 1.0]))
print("out:", out)
print("vjp grad:", grad_x)"""
    ),
    md(
        """## Gradient checkpointing

`jax.checkpoint` re-runs forward pieces during the backward pass instead of storing every activation. Same forward **value**, less memory, more compute."""
    ),
    code(
        """def deep_chain(x, depth=8):
    for _ in range(depth):
        x = jnp.tanh(x @ jnp.eye(x.shape[-1]))
    return jnp.sum(x)


x = jr.normal(jr.key(2), (32, 32))
checkpointed = jax.checkpoint(deep_chain)

loss_plain, _ = value_and_grad(deep_chain)(x)
loss_ckpt, _ = value_and_grad(checkpointed)(x)
print("same loss:", jnp.allclose(loss_plain, loss_ckpt))"""
    ),
    md(
        """> **Key insight:** Use `value_and_grad` for scalar training losses. Reach for `vjp` when the forward pass returns multiple outputs or you need a custom cotangent."""
    ),
    md(
        """---
## Exercise

1. Verify `grad` of `softmax(x) @ W` on a tiny example.
2. Implement CE loss + `value_and_grad`; compare to finite differences.
3. Call `jax.vjp` on a vector-valued function with a cotangent.
4. Wrap a deep function with `jax.checkpoint` and confirm the forward value is unchanged.

*(Solution below.)*"""
    ),
    code("print('loss plain:', float(loss_plain), '| checkpointed:', float(loss_ckpt))"),
    md("**Next:** Episode 4 — control flow with JIT."),
]

# EP04 is generated by scripts/build_ep04_control_flow.py

EP05 = [
    md(
        """# Episode 5 — Pytrees and SGD

**Instructor notebook** · run top-to-bottom before recording.

A 2-layer MLP as a nested **dict** of params; update every leaf with `tree_map` and a fixed learning rate.

| | |
|---|---|
| **Chapter** | 1.5 · Part I — Pure JAX |
| **Prereq** | Episodes 1–4 |
| **Next** | Part II — GPT-2 transformer |

**JAX docs:** [Pytrees](https://docs.jax.dev/en/latest/pytrees.html) · [`jax.tree.map`](https://docs.jax.dev/en/latest/_autosummary/jax.tree.map.html) · [`jax.tree.leaves`](https://docs.jax.dev/en/latest/_autosummary/jax.tree.leaves.html)
"""
    ),
    code(
        """import jax
import jax.numpy as jnp
import jax.random as jr"""
    ),
    md(
        """## Pytrees — nested dicts of arrays

JAX walks **nodes** (`dict`, `list`, `tuple`) and transforms **leaves** (arrays). Model weights are usually a nested dict — no custom classes needed."""
    ),
    md(
        """## 2-layer MLP as nested dict

`forward` takes `x` with a leading **batch** dimension `(B, D_in)` — same matmul + broadcast bias pattern from Episode 1. We do not use `vmap` here; the batch axis is part of the tensor shapes."""
    ),
    code(
        """def init_mlp(key, d_in, d_hidden, d_out):
    key, k0, k1 = jr.split(key, 3)
    return {
        0: {"w": jr.normal(k0, (d_in, d_hidden)) * 0.1, "b": jnp.zeros(d_hidden)},
        1: {"w": jr.normal(k1, (d_hidden, d_out)) * 0.1, "b": jnp.zeros(d_out)},
    }


def forward(params, x):
    # x: (B, D_in) — bias (D_hidden,) broadcasts over batch
    x = jnp.tanh(x @ params[0]["w"] + params[0]["b"])
    x = x @ params[1]["w"] + params[1]["b"]
    return x


key = jr.key(0)
params = init_mlp(key, 8, 16, 4)
B = 32
key, k_x = jr.split(key)
x_batch = jr.normal(k_x, (B, 8))
print("forward out:", forward(params, x_batch).shape)

leaves = jax.tree.leaves(params)
print(f"{len(leaves)} leaves:", [leaf.shape for leaf in leaves])"""
    ),
    md("## Inspect structure with `tree.map`"),
    code(
        """shapes = jax.tree.map(lambda a: a.shape, params)
print(shapes)"""
    ),
    md("## SGD — fixed learning rate, no optimizer state"),
    code(
        """LEARNING_RATE = 0.01


def sgd_update(params, grads):
    return jax.tree.map(lambda p, g: p - LEARNING_RATE * g, params, grads)


def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)


key, k_x, k_y = jr.split(jr.key(1), 3)
x_batch = jr.normal(k_x, (B, 8))
y_batch = jr.normal(k_y, (B, 4))
loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
params = sgd_update(params, grads)
print("batched loss after one step:", loss)"""
    ),
    md("## Training loop — return new params each step"),
    code(
        """def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = sgd_update(params, grads)
    return params, loss


params = init_mlp(jr.key(3), 8, 16, 4)
losses = []
for _ in range(20):
    params, loss = train_step(params, x_batch, y_batch)
    losses.append(float(loss))
print("first loss:", losses[0])
print("last loss: ", losses[-1])"""
    ),
    md("## `jit` the training step"),
    code(
        """jitted_step = jax.jit(train_step)
params = init_mlp(jr.key(4), 8, 16, 4)
params, loss = jitted_step(params, x_batch, y_batch)
print("jitted step loss:", loss)"""
    ),
    md(
        """> **Key insight:** Return **new** params every step — never mutate in place. Batch with a leading dimension on `x` and `y`; `tree_map` applies the same SGD rule to every leaf."""
    ),
    md(
        """---
## Exercise

1. Print all leaf shapes in the 2-layer MLP `params` dict.
2. Implement SGD with `tree_map` in two lines.
3. Run 20 batched steps on `(B, 8)` inputs; print the loss curve.

*(Solution below.)*"""
    ),
    code("print('final loss:', losses[-1])"),
    md("**Next:** Part II — build a GPT-2 transformer in pure JAX."),
]


def main() -> None:
    write_ep("ep01", EP01)
    write_ep("ep02", EP02)
    write_ep("ep03", EP03)
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_ep04_control_flow.py")],
        check=True,
        cwd=ROOT,
    )
    write_ep("ep05", EP05)
    print("done: ep01–ep05")


if __name__ == "__main__":
    main()
