# Episode 9 (draft) — Memory levers

Full implementation preserved from the original Ep8 scope. **Not recorded yet** — follows Ep8 memory accounting.

```bash
python ep09/deferred_memory_levers.py --find-max-batch
python ep09/deferred_memory_levers.py --profile
```

**Concepts:** `jax.checkpoint` · BF16 + FP32 master · `donate_argnums` · gradient accumulation

Builds on the memory intuition in [`ep08/gpt2.py`](../ep08/gpt2.py) — each lever shrinks a bucket you already counted.
