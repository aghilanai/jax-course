---
name: create-chapter
description: >-
  Scaffold or refresh JAX course chapter notebooks: solution.ipynb and
  student.ipynb. Use when creating a new episode/chapter, adding epNN folders,
  or when the user mentions create chapter or student workbook structure.
---

# Create Chapter

Each chapter lives in `epNN/` (e.g. `ep00`, `ep01`) and has **two notebooks**:

| File | Purpose |
|------|---------|
| `solution.ipynb` | Full instructor notebook — runnable code, exercise answers, extra notes. Check when you need help. |
| `student.ipynb` | Same markdown/text as `solution.ipynb`; every code cell is `# your code here`. What students fill in during the video. |

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

### 2. Create or refresh `student.ipynb`

Copy `solution.ipynb` to `student.ipynb`, then:

1. Replace the instructor header with `**Student workbook** · code along with the video.`
2. Remove `*(Solution below.)*` from exercise markdown cells.
3. For every code cell: set `source` to `["# your code here\n"]`, clear `outputs`, set `execution_count` to `null`.

**Warning:** refreshing overwrites `student.ipynb` and wipes any student edits.

### 3. Verify

- [ ] `solution.ipynb` runs top-to-bottom without errors
- [ ] `student.ipynb` has the same cell count and markdown as `solution.ipynb` (with student header)
- [ ] Every code cell in `student.ipynb` is only `# your code here`

## Directory layout

```
epNN/
├── solution.ipynb    # instructor — source of truth
└── student.ipynb     # text only, blank code
```

Shared course code (datasets, utils) stays in repo-level `data/`, not inside chapter folders.

## When the user asks to create a new chapter

1. Pick the next `epNN` number (check existing folders and `README.md`).
2. Create `epNN/solution.ipynb` with full content.
3. Create `epNN/student.ipynb` from the solution using the steps above.
4. Update `README.md` episode table if the chapter is ready to list.
