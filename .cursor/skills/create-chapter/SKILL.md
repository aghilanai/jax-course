---
name: create-chapter
description: >-
  Scaffold or refresh JAX course chapter notebooks: solution.ipynb,
  partial.ipynb, student.ipynb, and empty.ipynb. Use when creating a new
  episode/chapter, adding epNN folders, or when the user mentions
  create chapter, partial notebook, or student workbook structure.
---

# Create Chapter

Each chapter lives in `epNN/` (e.g. `ep00`, `ep01`) and has **four notebooks**:

| File | Purpose |
|------|---------|
| `solution.ipynb` | Full instructor notebook — runnable code, exercise answers, extra notes. Check when you need help. |
| `partial.ipynb` | Same markdown/text as `solution.ipynb`; every code cell is `# your code here`. |
| `student.ipynb` | Exact duplicate of `partial.ipynb` — what students fill in during the video. |
| `empty.ipynb` | One empty code cell, no markdown. Starting point for live coding on camera. |

## Workflow

### 1. Write `solution.ipynb` first

Author the complete episode in `solution.ipynb`:

- Markdown: title, prereqs, section headers, shape tables, exercises
- Code: imports, demos, exercise solutions
- Instructor header: `**Instructor notebook** · run top-to-bottom before recording.`
- Mark exercises with `*(Solution below.)*` before the answer cell when useful

Follow course conventions from existing chapters:

- No explicit `dtype=float32` unless necessary
- Capitalize shape letters in docs: `(N,)`, `(I, J)`, `(T,)`, `(V, V)`, `(S,)`
- Episode 0: simple names (`x`, `m`) — no Noam suffix notation until ML episodes
- Later episodes: suffix notation (`counts_ij`, `tokens_t`, etc.)

### 2. Generate the other three notebooks

From the repo root:

```bash
python .cursor/skills/create-chapter/scripts/create_chapter.py epNN
```

This writes:

- `partial.ipynb` — markdown preserved (student header), code stripped to placeholders, outputs cleared
- `student.ipynb` — copy of `partial.ipynb`
- `empty.ipynb` — blank notebook

Re-run after editing `solution.ipynb` to refresh `partial.ipynb` and `student.ipynb`.
**Warning:** re-running overwrites `student.ipynb` and wipes any student edits.

### 3. Verify

- [ ] `solution.ipynb` runs top-to-bottom without errors
- [ ] `partial.ipynb` / `student.ipynb` have matching cell count and markdown; code cells are only `# your code here`
- [ ] `empty.ipynb` has no markdown cells

## Manual creation (no script)

If not using the script:

**`partial.ipynb`** — copy `solution.ipynb`, then for every code cell replace `source` with `["# your code here\n"]` and clear `outputs`. Swap instructor header for `**Student workbook** · code along with the video.` Remove `*(Solution below.)*`.

**`student.ipynb`** — `cp partial.ipynb student.ipynb`

**`empty.ipynb`** — minimal notebook with one empty code cell:

```json
{
  "cells": [{"cell_type": "code", "metadata": {}, "source": [], "outputs": [], "execution_count": null}],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
  "nbformat": 4,
  "nbformat_minor": 5
}
```

## Directory layout

```
epNN/
├── solution.ipynb    # instructor — source of truth
├── partial.ipynb     # text only, blank code
├── student.ipynb     # duplicate of partial
└── empty.ipynb       # blank slate for recording
```

Shared course code (datasets, utils) stays in repo-level `data/`, not inside chapter folders.

## When the user asks to create a new chapter

1. Pick the next `epNN` number (check existing folders and `README.md`).
2. Create `epNN/solution.ipynb` with full content (or scaffold from `empty.ipynb` + outline).
3. Run `create_chapter.py epNN`.
4. Update `README.md` episode table if the chapter is ready to list.
