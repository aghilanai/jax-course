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


def studentize(notebook: dict) -> dict:
    out = deepcopy(notebook)
    for cell in out["cells"]:
        if cell["cell_type"] == "markdown":
            src = cell.get("source", [])
            text = "".join(src) if isinstance(src, list) else src
            text = text.replace(INSTRUCTOR, STUDENT)
            text = text.replace("*(Solution below.)*\n", "")
            text = text.replace("*(Solution below.)*", "")
            cell["source"] = [text] if text else src
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
    student.write_text(json.dumps(studentize(data), indent=1) + "\n")
    print(f"wrote {student}")


if __name__ == "__main__":
    main()
