#!/usr/bin/env python3
"""Generate partial.ipynb, student.ipynb, and empty.ipynb from solution.ipynb."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

PLACEHOLDER = "# your code here\n"
EMPTY_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [],
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _studentize_markdown(text: str) -> str:
    text = text.replace(
        "**Instructor notebook** · run top-to-bottom before recording.",
        "**Student workbook** · code along with the video.",
    )
    text = re.sub(r"\n\n\*\(Solution below\.\)\*\n?", "\n", text)
    return text


def partial_from_solution(solution: dict) -> dict:
    notebook = deepcopy(solution)
    notebook["cells"] = []

    for cell in solution["cells"]:
        if cell["cell_type"] == "markdown":
            source = cell.get("source", [])
            if isinstance(source, str):
                source = [source]
            text = "".join(source)
            cell = deepcopy(cell)
            cell["source"] = _studentize_markdown(text).splitlines(keepends=True)
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            notebook["cells"].append(cell)
            continue

        if cell["cell_type"] == "code":
            notebook["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": cell.get("metadata", {}),
                    "outputs": [],
                    "source": [PLACEHOLDER],
                }
            )

    return notebook


def write_notebook(path: Path, notebook: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def create_chapter(chapter_dir: Path, solution_name: str = "solution.ipynb") -> None:
    solution_path = chapter_dir / solution_name
    if not solution_path.exists():
        raise FileNotFoundError(f"Missing {solution_path}")

    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    partial = partial_from_solution(solution)

    partial_path = chapter_dir / "partial.ipynb"
    student_path = chapter_dir / "student.ipynb"
    empty_path = chapter_dir / "empty.ipynb"

    write_notebook(partial_path, partial)
    shutil.copyfile(partial_path, student_path)
    write_notebook(empty_path, deepcopy(EMPTY_NOTEBOOK))

    print(f"Wrote {partial_path}")
    print(f"Wrote {student_path} (copy of partial.ipynb)")
    print(f"Wrote {empty_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chapter_dir",
        type=Path,
        help="Chapter directory containing solution.ipynb (e.g. ep00)",
    )
    parser.add_argument(
        "--solution",
        default="solution.ipynb",
        help="Solution notebook filename (default: solution.ipynb)",
    )
    args = parser.parse_args()
    create_chapter(args.chapter_dir.resolve(), args.solution)


if __name__ == "__main__":
    main()
