"""Quick sanity check that JAX sees CUDA devices."""

import os

if os.environ.get("LD_LIBRARY_PATH"):
    print(
        "WARNING: LD_LIBRARY_PATH is set — this often breaks JAX CUDA in containers.\n"
        f"  LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}\n"
        "  Run: source .venv/bin/activate  (or use the JAX Course Jupyter kernel)"
    )

import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Devices ({len(jax.devices())}):")
for i, d in enumerate(jax.devices()):
    print(f"  [{i}] {d}")

x = jnp.ones((1024, 1024))
y = jnp.dot(x, x)
print(f"\nMatmul on {y.devices()}: shape={y.shape}, sum={float(y.sum()):.1f}")
