#!/usr/bin/env python3
"""Collect the values defined in the ``parameters`` cells of the recipe notebooks.

Static scan: nothing is executed, the parameters cell is parsed with ``ast``.
Values that are not literals (e.g. ``colors = DEFAULT_COLOR_NAMES``) are
reported as their source text and flagged ``unresolved``.

Usage (from the repo root)::

    python scripts/collect_parameters.py                     # json to stdout
    python scripts/collect_parameters.py -f yaml -o recipe_parameters.yaml
    python scripts/collect_parameters.py -f md > RECIPES.md
    python scripts/collect_parameters.py measurement/spot-detection   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recipe_params  # noqa: E402


def _flat(info: dict) -> dict:
    return {name: entry["value"] for name, entry in info["parameters"].items()}


def to_markdown(records: list[dict]) -> str:
    lines = ["| notebook | parameter | value |", "| --- | --- | --- |"]
    for rec in records:
        params = _flat(rec)
        if not params:
            continue
        first = True
        for name, value in params.items():
            label = rec["notebook"] if first else ""
            lines.append(f"| {label} | `{name}` | `{value!r}` |")
            first = False
    return "\n".join(lines) + "\n"


def to_yaml(records: list[dict]) -> str:
    """Nested ``notebook -> parameter -> value`` index.

    ``ruamel.yaml`` is an optional dependency (the repo already uses it in the
    downstream analysis notebooks).
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        sys.exit(
            "yaml output needs ruamel.yaml installed:\n"
            "  conda install -c conda-forge ruamel.yaml   or   pip install ruamel.yaml"
        )
    buf = StringIO()
    YAML(typ="safe").dump({rec["notebook"]: _flat(rec) for rec in records}, buf)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", default=[str(recipe_params.repo_root())],
                    help="directories/files to scan (default: repo root)")
    ap.add_argument("-f", "--format", choices=["json", "yaml", "md"], default="json")
    ap.add_argument("-o", "--out", default=None, help="write to this file instead of stdout")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--include-injected", action="store_true",
                    help="also pick up papermill's '# Injected parameters' cell")
    ap.add_argument("--summary", action="store_true", help="print a count per notebook to stderr")
    args = ap.parse_args()

    files: list[Path] = []
    for root in args.roots:
        root = Path(root).resolve()
        if root.is_file():
            files.append(root)
        else:
            files.extend(recipe_params.find_parameters_cell_notebooks(root, recursive=not args.no_recursive))

    records = []
    for p in sorted(files):
        info = recipe_params.parameters_from_notebook(p, include_injected=args.include_injected)
        rel = str(p.relative_to(recipe_params.repo_root())) if p.is_relative_to(recipe_params.repo_root()) else str(p)
        records.append({"notebook": rel, **{k: v for k, v in info.items() if k != "notebook"}})

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_commit": recipe_params._git_commit(),
        "notebook_count": len(records),
        "notebooks": {r["notebook"]: _flat(r) for r in records},
    }

    if args.format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    elif args.format == "yaml":
        text = to_yaml(records)
    else:
        text = to_markdown(records)

    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(records)} notebooks)", file=sys.stderr)
    else:
        sys.stdout.write(text)

    if args.summary:
        unresolved = {
            name
            for rec in records
            for name, entry in rec["parameters"].items()
            if entry.get("unresolved")
        }
        print(f"{len(records)} notebooks with a parameters cell", file=sys.stderr)
        if unresolved:
            print(f"non-literal (unresolved) values: {', '.join(sorted(unresolved))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
