# Episode 7 — Talking Script

**Notebook:** `solution.ipynb` · **Runtime:** ~12–18 min if you run cells live

Use this as a read-along while you scroll the notebook. `[RUN]` = execute the cell below.

---

## Intro

Welcome to Episode 7 — ML Optimizers.

Last episode we built the GPT and the training loop. This time **you** write the optimizers: SGD, Momentum, RMSProp, AdamW, and Muon — all as pure-JAX `NamedTuple`s with `init` and `__call__`.

The model, data, and `train(...)` live in `helpers.py`. Same loop for every optimizer; only the update rule changes.

Two links worth bookmarking: Distill on [why momentum works](https://distill.pub/2017/momentum/), and the [gradient optimizer comparison](https://www.corefranciscopark.com/blog/gradient-optimizer-comparison) for loss-landscape intuition.

---

## Setup `[RUN]`

Quick imports — JAX, our helpers, Muon utilities (`map_leaves`, `newton_schulz5`).

Everything else in this episode plugs into the same `train` function from Episode 6.

---

## Data + model `[RUN]`

Tiny Shakespeare again — about 338k tokens, 90/10 train/val split.

Same ~16M GPT as before: 4 layers, 256 hidden, 8 heads. Small enough to iterate fast, big enough to see optimizer differences.

---

## Optimizer contract

Every optimizer follows one interface:

```python
opt_state = optimizer.init(params)
params, opt_state = optimizer(params, grads, opt_state, step)
```

`train` only needs `.name`, `.learning_rate`, `.init`, and `__call__`. That's the whole plugin system — implement those four things and you're in the race.

---

## 1. SGD

**Read the formula:** θ ← θ − η·g.

Plain stochastic gradient descent. No memory, no state — just step opposite the gradient.

**Pros:** dead simple, strong baseline when the learning rate is tuned, often the best *final* loss if you're patient.

**Cons:** zigzags in narrow valleys, one global learning rate for every parameter, slow on ill-conditioned problems.

**Implement `SGD` `[RUN]`**

Three lines in `__call__`: tree-map over params and grads, subtract η times g. `init` returns `None` — no state.

This is our baseline. Everything after adds one idea on top.

---

## 2. Momentum (heavy-ball)

**Read the formula:** accumulate velocity v ← β·v + g, then θ ← θ − η·v.

Think of a ball rolling downhill. Consistent gradient directions build speed; oscillating directions cancel out. Great for skinny quadratic valleys — try β ≈ 0.99 in the Distill demo.

**Pros:** damps oscillation, wider usable learning-rate range than SGD.

**Cons:** extra state the size of the model, can overshoot sharp minima if β or η is too high, still no per-coordinate scaling.

**Implement `Momentum` `[RUN]`**

`init` allocates a zero velocity tree. Each step: update velocity, then step params along velocity — not along the raw gradient.

Notice the comment in the solution: velocity is really an exponential weighted sum of *all* past gradients, with older ones fading by β each step.

---

## 3. RMSProp

**Read the formula:** track v ← ρ·v + (1−ρ)·g², then scale the step by 1/√v.

Key insight: **square the gradient** so sign doesn't matter — we care about magnitude. Parameters with big recent gradients get smaller steps; quiet coordinates get larger ones.

**Pros:** per-coordinate step sizes, cheap (one extra state tree), good when gradient scales vary a lot.

**Cons:** no momentum on the gradient direction itself, still first-order and noisy, ρ and ε matter.

**Implement `RMSProp` `[RUN]`

Update the second-moment tree, then divide the gradient by √(moment + ε) before applying η.

This is adaptive SGD — same direction as g, different step size per parameter.

---

## 4. AdamW

**Read the full update:** first moment m (momentum), second moment v (RMSProp-style), bias-correct both, then step.

Two details that matter in practice:

1. **Bias correction** — m and v start at zero, so early steps would be tiny without dividing by (1 − β^t).
2. **Decoupled weight decay** — shrink θ by (1 − ηλ) *before* the Adam step. Don't fold L2 into the gradient; that's "Adam + L2" and it behaves differently.

**Pros:** default for transformers, robust learning rates, combines momentum + scaling + regularization.

**Cons:** 2× optimizer memory, diagonal only (ignores matrix structure), sensitive to mid-training LR changes — fix peak LR in short ablations before a long run.

**Implement `AdamW` `[RUN]`**

`AdamState` holds first and second moments. `step` is **1-indexed** to match `train`. Weight decay multiplies params down, then the bias-corrected Adam update subtracts.

This is the workhorse you'll reach for on embeddings, biases, and layernorm — and for everything if you're not using Muon.

---

## 5. Muon

The newest optimizer in the stack. For **2D weight matrices**: momentum on the gradient, then **Newton–Schulz orthogonalization** — project the update to the nearest semi-orthogonal matrix (think UVᵀ from the SVD). Scale by √(max(1, rows/cols)).

For **1D parameters** — embeddings, biases, vectors — fall back to Adam.

**Why?** Adam treats every scalar independently. Weight matrices have structure; Muon respects that with updates that have stable spectral norm. Reported wins on LLM pretraining speedruns.

**Pros:** strong sample efficiency on hidden layers, often less LR retuning when scaling width.

**Cons:** hybrid optimizer (Muon + Adam), extra matmuls per step, fewer battle-tested recipes, only applies to 2D leaves.

**Implement `Muon` `[RUN]`**

`map_leaves` branches on `grad.ndim == 2`. Matrix path: Nesterov momentum → `newton_schulz5` → scaled update. Vector path: standard AdamW logic with `adam_lr`.

Helpers do the heavy lifting — your job is wiring the branch and hyperparameters.

---

## Smoke test `[RUN]`

Fifty steps of AdamW on Tiny Shakespeare. Confirms your optimizer implements the contract and `train` accepts it.

Watch val loss tick down — if this runs, your AdamW is wired correctly.

---

## Curves + interactive demos

Show `optimizer_curves.png` — precomputed sweep on this corpus.

Loss curves tell you *who wins on this LM*; they don't show *why*. For geometry — Rosenbrock, narrow valleys, overshooting — use the [landscape demo](https://www.corefranciscopark.com/blog/gradient-optimizer-comparison). Momentum overshoots; Adam adapts; SGD zigzags.

---

## Takeaways

1. **SGD → Momentum** adds velocity — read Distill.
2. **RMSProp / AdamW** add per-coordinate scaling; AdamW adds proper weight decay.
3. **Muon** orthogonalizes matrix updates; 1D stays on Adam.
4. Same `train` loop for all — `init` + `__call__`, pass `optimizer=`.
5. Race them visually in the landscape demo.

---

## Exercise (optional on camera)

Three homework items:

1. Smoke-test `Muon()` for 50 steps — same `train(...)` call, swap the optimizer. `[RUN]` solution cell if you want to show it live.
2. In the landscape demo: Rosenbrock + all optimizers — who overshoots? (Usually high-momentum methods.)
3. From Distill: for fixed step size α, how should optimal β change as curvature λ grows?

---

## Outro

**Further reading:** Keller Jordan's [Muon post](https://kellerjordan.github.io/posts/muon/) for convergence diagrams, and NVIDIA's [SOAP, Muon, and Beyond](https://arxiv.org/pdf/2607.20548) for scaling pretraining.

Next episode: memory and mixed precision — making this training loop fit on one GPU.

---

## Quick reference (if you need a one-liner per optimizer)

| Optimizer | One line |
|-----------|----------|
| SGD | Step opposite the gradient. |
| Momentum | Smooth gradients into velocity; step along velocity. |
| RMSProp | Scale each coord by recent \|g\|. |
| AdamW | Momentum + scaling + bias fix + decoupled decay. |
| Muon | Orthogonal matrix updates; Adam for vectors. |
