# LLMs in JAX: Zero to Hero

Course materials for [**LLMs in JAX: Zero to Hero + Compute Analysis**](./syllabus.md).

## Setup

```bash
cd ~/Desktop/jax-course
pip install -r requirements.txt
```

Open this folder as the Cursor workspace (`File → Open Folder → jax-course`).

### Notebook IntelliSense

1. **Kernel** — *Select Kernel* → `.venv` (Python 3.12).
2. **Reload** — `Cmd+Shift+P` → *Developer: Reload Window* if completions are empty.
3. `.vscode/settings.json` points Pylance at `.venv` in this repo.

## Episodes

| Episode | Topic | Instructor | Student |
|---------|-------|------------|---------|
| 0 | Arrays, shapes, immutability | [solution](./ep00/solution.ipynb) | [student](./ep00/student.ipynb) |
| 1 | PRNG keys, `params`, broadcasting | [solution](./ep01/solution.ipynb) | [student](./ep01/student.ipynb) |
| 2 | `jit`, `vmap`, jaxpr, timing matmul | [solution](./ep02/solution.ipynb) | [student](./ep02/student.ipynb) |
| … | *see [syllabus](./syllabus.md)* | | |
| 10 | **Module 1 capstone** — [JAX Pytrees](https://docs.jax.dev/en/latest/pytrees.html) on GPT `params` | *planned* | *planned* |

**Workflow:** Karpathy-style — one idea per episode, short exercise at the end.

## Shared code

- `data/reflecting_digits.py` — spine dataset (Episode 5+).
