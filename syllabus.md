# ML Training Systems — Syllabus

**Source:** [Full lesson plan artifact](https://claude.ai/public/artifacts/326f2023-cecb-45db-959b-c97a870cafdf)

Train a transformer. Understand every byte. Five parts, 23 chapters, pure JAX from scratch — GPT-2 style architecture, GB200 FLOP accounting, real distributed training, scaling laws you fit yourself, pipeline and expert parallelism.

**Stack:** JAX · XLA · NCCL · 0 training libs

---

## Course map

| Part | Title | Chapters |
|------|-------|----------|
| **I** | Pure JAX | 5 |
| **II** | GPT-2 Transformer (no libs) | 5 |
| **III** | FLOP Accounting · GB200 · Sharding | 5 |
| **IV** | Scaling Laws · Chinchilla · MuP | 5 |
| **V** | Pipeline & Expert Parallelism | 3 |

**Notebooks:** `epNN/solution.ipynb` (instructor) and `epNN/student.ipynb` (code along). Part I maps `ep01`–`ep05` to Chapter 1.

---

## Part I — Pure JAX

### Foundation: The JAX Programming Model

5 chapters · no ML framework

| Episode | Section | Topic |
|---------|---------|-------|
| [1](./ep01/solution.ipynb) | 1.1 | JAX as a Functional Array Accelerator |
| [2](./ep02/solution.ipynb) | 1.2 | JIT, Tracing, and the Jaxpr |
| [3](./ep03/solution.ipynb) | 1.3 | Automatic Differentiation |
| [4](./ep04/solution.ipynb) | 1.4 | `vmap`, `scan`, and Vectorization |
| [5](./ep05/solution.ipynb) | 1.5 | Pytrees and SGD |

---

## Episode 1 — JAX as a Functional Array Accelerator

**Prerequisites:** Python, basic NumPy  
**Hardware:** CPU (GPU optional for timing exercises)

### Concepts

- Pure functions as a design constraint — no hidden state, no in-place mutation
- PRNG keys: `jax.random.key`, `split`, consume-on-use (no global RNG)
- `jnp` vs NumPy: same API surface, completely different execution model
- Broadcasting: `(B, D_out) + (D_out,)` bias add
- Device placement: `jax.devices()`, default device, explicit `jax.device_put()`
- Memory ownership — arrays live on device, copies are explicit
- Asynchronous dispatch: JAX returns futures, `block_until_ready()`
- Why JAX compiles rather than interprets: the case for deferred execution

### Exercises

- Write a matrix multiply in NumPy, port it to JAX, time both
- `split` a key; verify reproducible draws from the same root key
- Attempt an in-place mutation in JAX — observe the error
- Print `jax.devices()`; show `(B, D_out) + (D_out,)` with `jnp.broadcast_shapes`
- Benchmark `block_until_ready()` vs raw dispatch latency

### Key insight

The purity constraint is not a limitation — it is what makes compilation, differentiation, and parallelism composable. Every later chapter depends on this.

---

## Episode 2 — JIT, Tracing, and the Jaxpr

**Prerequisites:** Episode 1  
**Hardware:** CPU | Single GPU

### Concepts

- Tracing: JAX runs your Python function once with abstract `ShapedArray` values
- The JAX expression (jaxpr): a functional IR capturing the computation graph
- XLA HLO: what the jaxpr compiles down to, key ops (`dot`, `reduce`, `broadcast`)
- Compilation cache: when JIT reuses vs. retraces (shape changes, Python side-effects)
- **Static vs traced:** arrays traced by shape/dtype; Python scalars that control structure need `static_argnums`
- `jax.make_jaxpr()` — inspecting the trace before execution
- `jax.jit` with `static_argnums` for non-tensor arguments

### Exercises

- Print the jaxpr of a 3-layer MLP with `jax.make_jaxpr()`
- Trigger a retrace by changing input shape — count compilations with a counter inside the function
- Use `static_argnums` to pass a Python integer (e.g. `num_heads`) into a jitted function
- Measure wall-clock time: first JIT call (compile + run) vs. second call (run only)
- Write a function with a Python `print` inside — observe when it fires

### Key insight

JIT does not execute your Python code at runtime — it traces it once. Print statements inside jitted functions fire at trace time, not run time. This confusion catches everyone.

---

## Episode 3 — Automatic Differentiation

**Prerequisites:** Episodes 1–2  
**Hardware:** CPU | Single GPU

### Concepts

- `jax.grad`: scalar output only, returns a function
- `jax.vjp`: brief — cotangents when the forward pass is not scalar
- `jax.value_and_grad`: returns loss and gradient together (use this in training)
- Gradient checkpointing with `jax.checkpoint` — trade compute for memory

### Exercises

- Verify `grad` of `softmax(x) @ W` on a tiny example
- Implement cross-entropy loss with `value_and_grad`; check against finite differences
- Call `jax.vjp` on a vector-valued function with a cotangent
- Wrap a deep function with `jax.checkpoint` and confirm the forward value is unchanged

### Key insight

Use `value_and_grad` for scalar training losses. Reach for `vjp` when the forward pass returns multiple outputs or you need a custom cotangent.

---

## Episode 4 — `vmap`, `scan`, and Vectorization

**Prerequisites:** Episodes 1–3  
**Hardware:** CPU | Single GPU

### Concepts

- XLA vectorization preview — why batching beats Python loops
- `jax.vmap`: lifts a per-sample function to a batched function
- `in_axes`: share `params`, batch over inputs
- `jit` ∘ `vmap`: one compiled batched kernel
- `jax.lax.scan` vs Python `for` — prefix sum as carry loop
- Timing: loop vs `vmap` vs `scan`

### Exercises

- `vmap` a single-sample forward; verify against a Python loop
- Time `vmap` vs loop on a large batch
- Implement inclusive prefix sum with `scan`; compare to a Python `for` loop

### Key insight

`vmap` is taught once here. Later episodes batch with a leading tensor dimension instead.

---

## Episode 5 — Pytrees and SGD

**Prerequisites:** Episodes 1–4  
**Hardware:** CPU | Single GPU

### Concepts

- Pytrees: nested `dict` of arrays — nodes vs leaves
- `jax.tree.leaves`, `jax.tree.map`
- 2-layer MLP `params` as `{0: {"w", "b"}, 1: {"w", "b"}}`
- Batched forward: `x` shape `(B, D_in)` with broadcasting bias (no `vmap`)
- Functional update: return new params, never mutate
- SGD with a fixed learning rate — no optimizer state

### Exercises

- Print all leaf shapes in a 2-layer MLP `params` dict
- Implement SGD update with `tree_map` in two lines
- Run 20 batched training steps on `(B, D_in)` inputs; print the loss curve
- `jit` a `train_step` that returns new params

### Key insight

Return new params every step — never mutate in place. Batch with a leading dimension on `x` and `y`; a fixed learning rate means `tree_map` is all you need for SGD.

---

## Parts II–V (planned)

| Part | Focus |
|------|-------|
| **II** | GPT-2 transformer from scratch — embeddings, attention, blocks, training loop, generation |
| **III** | FLOP models, MFU, `jax.profiler`, `Mesh`, `PartitionSpec`, collectives |
| **IV** | Scaling laws, Chinchilla, MuP, hyperparameter transfer |
| **V** | Pipeline parallelism, expert parallelism (conceptual + worked examples) |

---

## Dependency graph (Part I)

```mermaid
flowchart LR
    E1[Ep1 Arrays] --> E2[Ep2 JIT]
    E2 --> E3[Ep3 AD]
    E3 --> E4[Ep4 vmap/scan]
    E4 --> E5[Ep5 Pytrees]
    E5 --> P2[Part II Transformer]
```
