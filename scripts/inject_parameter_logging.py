#!/usr/bin/env python3
"""Insert (or refresh) an auto-managed parameter-logging cell in every notebook
that has a cell tagged ``parameters``.

The cell is placed directly *after* the parameters cell, which is exactly where
papermill injects its override cell - so the managed cell runs *after* the
overrides and records the values that were actually used.  When you run a
notebook by hand in Jupyter the same cell simply records the defaults that are
currently in the file.

Usage (from the repo root)::

    python scripts/inject_parameter_logging.py                                  # add/refresh everywhere
    python scripts/inject_parameter_logging.py resave/resave_nd2_as_tiff.ipynb  # just one recipe
    python scripts/inject_parameter_logging.py segmentation/ spot-detection/    # these subtrees (recursive)
    python scripts/inject_parameter_logging.py --check                          # report drift only
    python scripts/inject_parameter_logging.py --remove                         # drop managed cells
    python scripts/inject_parameter_logging.py --remove segmentation/           # drop them in a subtree only

Positional ``roots`` restrict the run to directories and/or individual
notebooks; ``--check`` and ``--remove`` take the same arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recipe_params  # noqa: E402

HEADER = "<< auto-managed by scripts/inject_parameter_logging.py - do not edit >>"

TEMPLATE = """# {header}
import sys, pathlib
for _p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
    if (_p / 'recipe_params.py').is_file():
        sys.path.insert(0, str(_p))
        break
import recipe_params
recipe_params.capture(globals(), notebook={notebook})
"""

def cell_source(notebook: str) -> str:
    return TEMPLATE.format(header=HEADER, notebook=repr(notebook))


def managed_cell_metadata(notebook: str) -> dict:
    return {
        "tags": ["parameter-capture"],
        "editable": True,
        "slideshow": {"slide_type": ""},
        recipe_params.MANAGED_KEY: {"managed": recipe_params.MANAGED_VALUE, "notebook": notebook},
    }


def is_managed(cell) -> bool:
    return (cell.get("metadata") or {}).get(recipe_params.MANAGED_KEY, {}).get("managed") == recipe_params.MANAGED_VALUE


def update_notebook(path: Path, repo_root: Path, remove: bool = False) -> str:
    """Returns 'added', 'updated', 'removed', 'unchanged' or 'no-parameters-cell'."""
    nb = nbformat.read(path, as_version=4)
    rel = str(path.relative_to(repo_root))
    wanted = cell_source(rel)

    idx = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code" and recipe_params._is_parameters_cell(c)), None)
    if idx is None:
        return "no-parameters-cell"

    following = idx + 1
    existing = following if following < len(nb.cells) and is_managed(nb.cells[following]) else None

    if remove:
        if existing is None:
            return "unchanged"
        del nb.cells[existing]
        status = "removed"
    elif existing is not None:
        cell = nb.cells[existing]
        if cell.source == wanted and cell.metadata == managed_cell_metadata(rel):
            return "unchanged"
        cell.source = wanted
        cell.metadata = managed_cell_metadata(rel)
        status = "updated"
    else:
        cell = nbformat.v4.new_code_cell(source=wanted, metadata=managed_cell_metadata(rel))
        nb.cells.insert(following, cell)
        status = "added"

    nbformat.write(nb, path)
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", default=[str(recipe_params.repo_root())])
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    repo_root = recipe_params.repo_root()
    files: list[Path] = []
    for root in args.roots:
        root = Path(root).resolve()
        if root.is_file():
            files.append(root)
        else:
            files.extend(recipe_params.find_parameters_cell_notebooks(root))

    counts: dict[str, int] = {}
    for p in sorted(set(files)):
        rel = str(p.relative_to(repo_root))
        if args.check:
            nb = nbformat.read(p, as_version=4)
            idx = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code" and recipe_params._is_parameters_cell(c)), None)
            if idx is None:
                status = "no-parameters-cell"
            else:
                nxt = nb.cells[idx + 1] if idx + 1 < len(nb.cells) else None
                if nxt is None or not is_managed(nxt):
                    status = "missing"
                elif nxt.source != cell_source(rel):
                    status = "outdated"
                else:
                    status = "unchanged"
        else:
            status = update_notebook(p, repo_root, remove=args.remove)
        counts[status] = counts.get(status, 0) + 1
        if status != "unchanged":
            print(f"{status}: {rel}")

    print(f"\n{counts}", file=sys.stderr)
    return 1 if args.check and any(s in counts for s in ("missing", "outdated")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
