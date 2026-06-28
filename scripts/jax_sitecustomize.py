"""Installed into .venv by scripts/setup.sh — runs before user code on every Python start."""

import os

# Pip-bundled CUDA libs conflict with /usr/local/cuda/compat in LD_LIBRARY_PATH.
os.environ.pop("LD_LIBRARY_PATH", None)
# Silence XLA GPU autotuning warnings on first JIT compile.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
