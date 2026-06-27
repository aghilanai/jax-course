# ML Training Systems — Pure JAX (Part I)

Course materials for [**ML Training Systems**](./syllabus.md) — train a transformer, understand every byte.

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

## Part I — Chapter 1: The JAX Programming Model

| Episode | Topic | Instructor | Student |
|---------|-------|------------|---------|
| 1 | JAX as a Functional Array Accelerator | [solution](./ep01/solution.ipynb) | [student](./ep01/student.ipynb) |
| 2 | JIT, Tracing, and the Jaxpr | [solution](./ep02/solution.ipynb) | [student](./ep02/student.ipynb) |
| 3 | Automatic Differentiation | [solution](./ep03/solution.ipynb) | [student](./ep03/student.ipynb) |
| 4 | Control Flow with JIT | [solution](./ep04/solution.ipynb) | [student](./ep04/student.ipynb) |
| 5 | Pytrees and SGD | [solution](./ep05/solution.ipynb) | [student](./ep05/student.ipynb) |

**Workflow:** Karpathy-style — one idea per episode, short exercise at the end.

### Regenerate student notebooks

After editing `solution.ipynb`:

```bash
python scripts/build_student.py ep01
```

Or rebuild all of Chapter 1 from the generator:

```bash
python scripts/build_chapter1.py
```

## Syllabus source

Full five-part plan: [Claude artifact](https://claude.ai/public/artifacts/326f2023-cecb-45db-959b-c97a870cafdf)
