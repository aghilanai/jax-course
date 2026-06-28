#!/usr/bin/env bash
# Create venv, install JAX with CUDA, register Jupyter kernel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Unset LD_LIBRARY_PATH before Python starts (container cuda/compat breaks pip JAX CUDA).
if [ ! -f .venv/bin/python.real ]; then
  mv .venv/bin/python .venv/bin/python.real
  install -m 755 "$ROOT/scripts/python-wrapper" .venv/bin/python
fi

grep -q 'unset LD_LIBRARY_PATH' .venv/bin/activate 2>/dev/null || cat >> .venv/bin/activate <<'EOF'

# Pip-bundled CUDA libs conflict with /usr/local/cuda/compat in LD_LIBRARY_PATH.
unset LD_LIBRARY_PATH
EOF

install -m 755 "$ROOT/scripts/jupyter-kernel" .venv/bin/jupyter-kernel

.venv/bin/python.real -m ipykernel install --user --name=jax-course --display-name="JAX Course (.venv)"
KERNEL_DIR="$(.venv/bin/python.real -m jupyter kernelspec list --json | .venv/bin/python.real -c "import json,sys; print(json.load(sys.stdin)['kernelspecs']['jax-course']['resource_dir'])")"
.venv/bin/python.real - "$KERNEL_DIR" "$ROOT" <<'PY'
import json, sys
path = f"{sys.argv[1]}/kernel.json"
root = sys.argv[2]
spec = {
    "argv": [f"{root}/.venv/bin/jupyter-kernel", "-f", "{connection_file}"],
    "display_name": "JAX Course (.venv)",
    "language": "python",
    "metadata": {"debugger": True},
}
with open(path, "w") as f:
    json.dump(spec, f, indent=1)
    f.write("\n")
PY

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Verify GPUs: python scripts/verify_jax_devices.py"
