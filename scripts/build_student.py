#!/usr/bin/env python3
"""Generate student.ipynb from solution.ipynb."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

PLACEHOLDER = ["# your code here\n"]
INSTRUCTOR = "**Instructor notebook** · run top-to-bottom before recording."
STUDENT = "**Student workbook** · code along with the video."


def _markdown_text(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def studentize(notebook: dict, existing: dict | None = None) -> dict:
    out = deepcopy(notebook)
    existing_cells = existing.get("cells", []) if existing else []
    same_layout = len(existing_cells) == len(out["cells"])
    for i, cell in enumerate(out["cells"]):
        if cell["cell_type"] == "markdown":
            if (
                same_layout
                and existing_cells[i]["cell_type"] == "markdown"
            ):
                cell["source"] = deepcopy(existing_cells[i]["source"])
                continue
            text = _markdown_text(cell)
            text = text.replace(INSTRUCTOR, STUDENT)
            text = text.replace("*(Solution below.)*\n", "")
            text = text.replace("*(Solution below.)*", "")
            cell["source"] = [text] if text else cell.get("source", [])
        elif cell["cell_type"] == "code":
            cell["source"] = PLACEHOLDER.copy()
            cell["outputs"] = []
            cell["execution_count"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path, help="e.g. ep01")
    args = parser.parse_args()
    ep = args.episode
    solution = ep / "solution.ipynb"
    student = ep / "student.ipynb"
    data = json.loads(solution.read_text())
    existing = json.loads(student.read_text()) if student.exists() else None
    student.write_text(json.dumps(studentize(data, existing), indent=1) + "\n")
    print(f"wrote {student}")


if __name__ == "__main__":
    main()
