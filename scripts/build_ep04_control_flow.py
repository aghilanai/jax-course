#!/usr/bin/env python3
"""Generate ep04/solution.ipynb from JAX control-flow guide content."""

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
        "source": [text if text.endswith("\n") else text + "\n"],
    }


CELLS = [
    md(
        """# Episode 4 — Control Flow with JIT

**Instructor notebook** · run top-to-bottom before recording.

Python `if` / `for` / `while` work eagerly, but **`jit` traces one path** through your control-flow graph. When the path depends on **input values**, you need `static_argnames`, `lax.cond`, or `lax.*_loop`.

| | |
|---|---|
| **Chapter** | 1.4 · Part I — Pure JAX |
| **Prereq** | Episodes 1–3 |
| **Next** | Episode 5 — pytrees and SGD |

**Source:** [Control flow and logical operators with JIT](https://docs.jax.dev/en/latest/control-flow.html)"""
    ),
    code(
        """from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, lax"""
    ),
    md(
        """## Control flow under `jit`

Eagerly, JAX behaves like NumPy. Under **`jit`**, Python control flow is evaluated at **compile time** — the compiled function is **one path** through the graph. Logical operators short-circuit the same way.

If the path depends on **input values**, tracing fails by default. It may depend on **shape/dtype** — then JAX **recompiles** on new shapes."""
    ),
    md("### What works"),
    code(
        """@jit
def f(x):
    for i in range(3):  # compile-time constant — always 3 trips
        x = 2 * x
    return x


print(f(3))"""
    ),
    code(
        """@jit
def g(x):
    y = 0.0
    for i in range(x.shape[0]):  # trip count from shape, not value
        y = y + x[i]
    return y


print(g(jnp.array([1.0, 2.0, 3.0])))"""
    ),
    md("### What fails (value-dependent branch)"),
    code(
        """@jit
def f(x):
    if x < 3:
        return 3.0 * x ** 2
    else:
        return -4 * x


try:
    f(2)
except jax.errors.TracerBoolConversionError as e:
    print("TracerBoolConversionError:", e)"""
    ),
    code(
        """@jit
def g(x):
    return (x > 0) and (x < 3)


try:
    g(2)
except jax.errors.TracerBoolConversionError as e:
    print("TracerBoolConversionError:", e)"""
    ),
    md(
        """### Why?

`jit` traces on **`ShapedArray`** — shape + dtype, not a specific value. For `if x < 3`, the predicate is `{True, False}` abstractly; Python cannot pick a branch, so tracing stops.

**Tradeoff:** more abstract traces → fewer recompiles, but stricter rules on Python control flow."""
    ),
    md("### `static_argnames` — trace on concrete values"),
    code(
        """def f(x):
    if x < 3:
        return 3.0 * x ** 2
    else:
        return -4 * x


f_static = jit(f, static_argnames="x")
print(f_static(2.0))"""
    ),
    code(
        """def sum_first_n(x, n):
    y = 0.0
    for i in range(n):
        y = y + x[i]
    return y


sum_first_n_jit = jit(sum_first_n, static_argnames="n")
print(sum_first_n_jit(jnp.array([2.0, 3.0, 4.0]), 2))"""
    ),
    md(
        """With `static_argnames='n'`, the loop is **statically unrolled** at trace time (fine for small `n`; disastrous if `n` changes every call).

⚠️ **`static_argnames` can be handy when `length` rarely changes, but costly if it changes a lot.**"""
    ),
    md(
        """### Value-dependent shapes

Same issue when **array shapes** depend on argument **values** (shape-specialization on dtype/shape alone is OK):"""
    ),
    code(
        """def example_fun(length, val):
    return jnp.ones((length,)) * val


print(example_fun(5, 4))"""
    ),
    code(
        """bad_example_jit = jit(example_fun)

try:
    bad_example_jit(10, 4)
except TypeError as e:
    print("TypeError:", e)"""
    ),
    code(
        """good_example_jit = jit(example_fun, static_argnames="length")
print(good_example_jit(10, 4))
print(good_example_jit(5, 4))  # recompiles — new output shape"""
    ),
    md(
        """### Side effects inside `jit`

`print` runs at trace time and shows **tracers**, not concrete values. For debug output inside compiled code, use [`jax.debug.print`](https://docs.jax.dev/en/latest/_autosummary/jax.debug.print.html) (Episode 2)."""
    ),
    code(
        """@jit
def f(x):
    print("x:", x)
    y = 2 * x
    print("y:", y)
    return y


f(2)"""
    ),
    md(
        """## Structured control flow primitives

When you want **traceable** control flow **without** recompiling on every branch — and **without** unrolling large loops — use `jax.lax`:

| Primitive | Role |
|---|---|
| `lax.cond` | differentiable `if` on a scalar predicate |
| `lax.while_loop` | `while` with runtime stop condition |
| `lax.fori_loop` | counted `for`; XLA may lower to `scan` or `while_loop` |
| `lax.scan` | fold/map/scan with per-step inputs |"""
    ),
    md("### `lax.cond`"),
    code(
        """operand = jnp.array([0.0])
print(lax.cond(True, lambda x: x + 1, lambda x: x - 1, operand))
print(lax.cond(False, lambda x: x + 1, lambda x: x - 1, operand))"""
    ),
    md(
        """Related: [`lax.select`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.select.html) (batched choices as arrays), [`lax.switch`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html) (multi-branch). NumPy-style: `jnp.where`, `jnp.piecewise`, `jnp.select`."""
    ),
    md("### `lax.while_loop`"),
    code(
        """init_val = 0
cond_fun = lambda x: x < 10
body_fun = lambda x: x + 1
print(lax.while_loop(cond_fun, body_fun, init_val))"""
    ),
    md("### `lax.fori_loop`"),
    code(
        """init_val = 0
body_fun = lambda i, x: x + i
print(lax.fori_loop(0, 10, body_fun, init_val))"""
    ),
    md(
        """### Summary — `jit` vs `grad`

| construct | `jit` | `grad` |
|---|---|---|
| `if` | ❌ (value-dependent) | ✔ |
| `for` | ✔* | ✔ |
| `while` | ✔* | ✔ |
| `lax.cond` | ✔ | ✔ |
| `lax.while_loop` | ✔ | fwd only |
| `lax.fori_loop` | ✔ | fwd only† |
| `lax.scan` | ✔ | ✔ |

\\* loop bound must be **value-independent** (compile-time / shape-based) — otherwise unrolls or fails.

† `fori_loop` is rev-mode differentiable when endpoints are static."""
    ),
    md(
        """## Logical operators

Use **`jnp.logical_and` / `logical_or` / `logical_not`** (or bitwise `&` `|` `~`) under `jit`. Unlike Python `and`/`or`, they **do not short-circuit** — both sides are evaluated."""
    ),
    code(
        """def python_check_positive_even(x):
    is_even = x % 2 == 0
    return is_even and (x > 0)


@jit
def jax_check_positive_even(x):
    is_even = x % 2 == 0
    return jnp.logical_and(is_even, x > 0)


print(python_check_positive_even(24))
print(jax_check_positive_even(24))"""
    ),
    code(
        """x = jnp.array([-1, 2, 5])
print(jax_check_positive_even(x))"""
    ),
    code(
        """try:
    python_check_positive_even(x)
except ValueError as e:
    print("ValueError:", e)"""
    ),
    md(
        """## Python control flow + `grad` (no `jit`)

The constraints above apply to **`jit` only**. **`grad`** on pure Python functions with `if`/`for` works fine — Autograd-style:"""
    ),
    code(
        """def f(x):
    if x < 3:
        return 3.0 * x ** 2
    else:
        return -4 * x


print(grad(f)(2.0))
print(grad(f)(4.0))"""
    ),
    md(
        """> **Key insight:** Under `jit`, value-dependent Python control flow fails or forces recompilation via `static_argnames`. For dynamic branches and loops inside one compiled graph, use **`lax.cond`**, **`lax.while_loop`**, **`lax.fori_loop`**, or **`lax.scan`**."""
    ),
    md(
        """---
## Exercise

1. Write a `@jit` function with `if x < 0` that fails; fix it with `static_argnames` or `lax.cond`.
2. Write `example_fun(length, val)` with `jit` and `static_argnames='length'`; call it with two different lengths.
3. Implement a counted sum with `lax.fori_loop` and a runtime stop with `lax.while_loop`.
4. Replace `(x > 0) and (x < 3)` with `jnp.logical_and` under `jit`.

*(Solution below.)*"""
    ),
    code(
        """@jit
def abs_scaled(x):
    return lax.cond(x < 0, lambda v: -2.0 * v, lambda v: 2.0 * v, x)


print("cond:", abs_scaled(-3.0), abs_scaled(3.0))
print("shapes:", good_example_jit(3, 1.0).shape, good_example_jit(7, 1.0).shape)
print("fori:", lax.fori_loop(0, 5, lambda i, a: a + i, 0))
print("while:", lax.while_loop(lambda s: s < 5, lambda s: s + 1, 0))
print("logical:", jax_check_positive_even(jnp.array([2, 4, -2])))"""
    ),
    md("**Next:** Episode 5 — pytrees and SGD."),
]


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "JAX Course (.venv)",
                "language": "python",
                "name": "jax-course",
            },
            "language_info": {"name": "python"},
        },
        "cells": CELLS,
    }
    path = ROOT / "ep04" / "solution.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
