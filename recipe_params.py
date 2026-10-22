"""Parameter capture helpers for the recipe-book notebooks.

Two entry points:

``parameters_from_notebook(path)``
    Static extraction of the assignments in the cell tagged ``parameters``
    (no execution needed, works on plain files on disk).

``capture(globals(), notebook=...)``
    Dump the *effective* parameter values of a notebook that is currently
    running. Because it reads the values out of the live kernel namespace it
    works identically for manual Jupyter runs and papermill/nbclient runs
    (papermill's injected cell has already overwritten the defaults by the
    time the capture cell runs, since it is injected *after* the parameters
    cell and *before* the capture cell).

Both write plain JSON, see ``RECIPE_*`` environment variables below for
overriding where the dumps go.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PARAMETERS_TAG = "parameters"
INJECTED_TAG = "injected-parameters"
#: papermill prefixes its injected cell with a comment; older versions used
#: "# Injected parameters", current versions use "# Parameters"
INJECTED_COMMENTS = ("# Parameters", "# Injected parameters")

#: Environment variables that override the defaults (set them before running
#: the wrapper notebook if you want the dumps somewhere else).
ENV_REPO_ROOT = "RECIPE_REPO_ROOT"
ENV_NOTEBOOK = "RECIPE_NOTEBOOK"
ENV_OUT_DIR = "RECIPE_PARAM_OUT_DIR"

#: Cell metadata marker used by scripts/inject_parameter_logging.py
MANAGED_KEY = "recipe_book"
MANAGED_VALUE = "parameter-logging"

#: provenance file written next to the results, so a dataset keeps a record of
#: the parameters it was processed with
OUTPUT_PROVENANCE_NAME = "recipe_parameters.json"


# --------------------------------------------------------------------------
# reading parameters out of a notebook file
# --------------------------------------------------------------------------
def _is_parameters_cell(cell) -> bool:
    return PARAMETERS_TAG in (cell.get("metadata") or {}).get("tags", [])


def _is_injected_cell(cell) -> bool:
    """papermill injects an explicit cell right after the parameters cell."""
    if cell.get("cell_type") != "code":
        return False
    tags = (cell.get("metadata") or {}).get("tags", [])
    if INJECTED_TAG in tags:
        return True
    first_line = "".join(cell.get("source", "")).strip().splitlines()
    return bool(first_line) and first_line[0].strip() in INJECTED_COMMENTS


_UNRESOLVED = object()


def _literal(node):
    """Best-effort literal value for an AST node; ``_UNRESOLVED`` if not a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _UNRESOLVED


def _assignments(source: str) -> dict:
    """Collect ``name -> value`` for the assignments in a parameters cell.

    Later assignments win (so ``x = 1`` followed by ``x = 2`` reports 2, the
    value the notebook actually uses).  Values that cannot be evaluated as
    literals (e.g. ``colors = DEFAULT_COLOR_NAMES``) are reported as their
    source text with ``unresolved`` set.
    """
    params: dict = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return params

    # sorted by position so source order (and therefore "last assignment wins")
    # is preserved even when assignments sit inside if/for blocks
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        if isinstance(node, ast.Assign):
            targets, value_node = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value_node = [node.target], node.value
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            targets, value_node = [node.target], node.value
        else:
            continue

        value = _literal(value_node)
        raw = ast.get_source_segment(source, value_node)

        # unpacking, e.g. z_name, y_name, x_name = "z_um", "y_um", "x_um"
        if len(targets) == 1 and isinstance(targets[0], ast.Tuple) and value is not _UNRESOLVED and isinstance(value, tuple):
            names = [t.id for t in targets[0].elts if isinstance(t, ast.Name)]
            if len(names) == len(value):
                for name, v in zip(names, value):
                    params[name] = {"value": v}
                continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if value is _UNRESOLVED:
                params[target.id] = {"value": raw, "unresolved": True}
            else:
                params[target.id] = {"value": value}

    return params


def parameters_from_notebook(path, include_injected: bool = True) -> dict:
    """Read the parameter defaults (and, if present, papermill's injected
    overrides) from a notebook file.

    Returns ``{"notebook": str, "parameters": {name: {"value": ...}}, ...}``.
    """
    path = Path(path)
    nb = json.loads(path.read_text())

    defaults: dict = {}
    injected: dict = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if _is_parameters_cell(cell):
            defaults = _assignments("".join(cell.get("source", "")))
        elif include_injected and _is_injected_cell(cell):
            injected = _assignments("".join(cell.get("source", "")))

    # injected parameters win over the defaults written in the file
    effective = {**defaults, **injected}

    return {
        "notebook": str(path),
        "parameters": effective,
        "defaults": defaults,
        "injected": injected,
    }


def _safe_parameters(path, include_injected: bool = True) -> dict:
    """``parameters_from_notebook`` that never raises (corrupt/odd notebooks are
    simply treated as having no parameters)."""
    try:
        return parameters_from_notebook(path, include_injected=include_injected)
    except (OSError, ValueError, KeyError):
        return {"notebook": str(path), "parameters": {}, "defaults": {}, "injected": {}}


def find_parameters_cell_notebooks(root, recursive: bool = True) -> list[Path]:
    """All notebooks below ``root`` that have a cell tagged ``parameters``."""
    root = Path(root)
    pattern = "**/*.ipynb" if recursive else "*.ipynb"
    found = []
    for p in sorted(root.glob(pattern)):
        if ".ipynb_checkpoints" in p.parts:
            continue
        try:
            nb = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if any(_is_parameters_cell(c) for c in nb.get("cells", [])):
            found.append(p)
    return found


# --------------------------------------------------------------------------
# runtime capture
# --------------------------------------------------------------------------
def _to_jsonable(value):
    """Convert common scientific values to something json.dump can handle."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    try:  # numpy scalars / arrays, without importing numpy eagerly
        if hasattr(value, "tolist"):
            return _to_jsonable(value.tolist())
        item = getattr(value, "item", None)
        if callable(item):
            return _to_jsonable(item())
    except Exception:
        pass
    return {"repr": repr(value), "unresolved": True}


def repo_root(start: Path | None = None) -> Path:
    """Repo root: ``RECIPE_REPO_ROOT`` if set, else the nearest directory up
    from ``start``/cwd that contains ``recipe_params.py``."""
    if env := os.environ.get(ENV_REPO_ROOT):
        return Path(env)
    start = start or Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "recipe_params.py").is_file():
            return candidate
    return start


#: alias for older callers
_repo_root = repo_root


def _detect_notebook(globals_: dict, cwd: Path, repo_root: Path) -> Path | None:
    """Guess which notebook we are running in.

    Cascade: explicit ``RECIPE_NOTEBOOK`` env var -> ``__file__`` (never set in
    notebooks, but harmless) -> match the live namespace against the parameters
    cells of the notebooks in the current directory.
    """
    if env := os.environ.get(ENV_NOTEBOOK):
        return (repo_root / env).resolve()
    if file := globals_.get("__file__"):
        return Path(file).resolve()

    candidates = [p for p in repo_root.glob("**/*.ipynb") if ".ipynb_checkpoints" not in p.parts]
    best, best_score = None, 0
    for p in candidates:
        names = _safe_parameters(p)["parameters"]
        if not names:
            continue
        score = 0
        for name, entry in names.items():
            if name not in globals_:
                continue
            live = _to_jsonable(globals_[name])
            stored = _to_jsonable(entry["value"])
            score += 1 if live == stored or entry.get("unresolved") else 0
        if score > best_score:
            best, best_score = p, score
    return best if best_score else None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def capture(
    globals_: dict,
    notebook: str | Path | None = None,
    out_dir: str | Path | None = None,
    extra: dict | None = None,
    write_latest: bool = True,
    append_log: bool = True,
    also_to_output_dir: bool = True,
    base_dir_key: str = "in_path",
    output_dir_keys: tuple[str, ...] = (
        "out_subdirectory", "out_subdir", "out_subfolder", "output_subdirectory",
    ),
    verbose: bool = True,
) -> dict:
    """Dump the parameter values of a running notebook to JSON.

    ``globals_`` should be ``globals()`` from inside the notebook.  Names are
    taken from the notebook's ``parameters`` cell when the notebook can be
    located; otherwise every JSON-serialisable name in the namespace is
    recorded (which is still a useful provenance record).

    Writes ``<out_dir>/<notebook-stem>.json`` (latest run) and appends a line
    to ``<out_dir>/runs.jsonl`` (append-only log of every run).  ``also_to_output_dir``
    defaults to True: a copy is written into the directory the notebook puts its
    results in (``in_path`` + ``out_subdirectory`` &co), so a dataset carries its
    own provenance record.
    """
    root = repo_root()
    nb_path = (root / notebook).resolve() if notebook else _detect_notebook(globals_, Path.cwd(), root)

    record = {
        "notebook": str(nb_path.relative_to(root)) if nb_path else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd()),
        "git_commit": _git_commit(),
        # set RECIPE_RUNNER=papermill in the wrapper notebook to label runs
        "runner": os.environ.get("RECIPE_RUNNER", "interactive"),
        "parameters": {},
        "changed": {},
    }

    file_defaults = {}
    if nb_path and nb_path.suffix == ".ipynb" and nb_path.is_file():
        info = _safe_parameters(nb_path, include_injected=False)
        file_defaults = {k: _to_jsonable(v["value"]) for k, v in info["defaults"].items()}
        names = list(info["defaults"])
    else:  # fall back: everything in the namespace that can be serialised
        names = [n for n, v in globals_.items() if not n.startswith("_") and not callable(v)]

    for name in names:
        if name not in globals_:
            continue
        value = _to_jsonable(globals_[name])
        record["parameters"][name] = value
        if name in file_defaults and file_defaults[name] != value:
            record["changed"][name] = {"default": file_defaults[name], "used": value}

    if extra:
        record["parameters"].update({k: _to_jsonable(v) for k, v in extra.items()})

    out_dir = Path(out_dir or os.environ.get(ENV_OUT_DIR) or root / "parameter_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if write_latest:
        stem = Path(record["notebook"]).stem if record["notebook"] else "unnamed"
        (out_dir / f"{stem}.json").write_text(payload)
    if append_log:
        with open(out_dir / "runs.jsonl", "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    if also_to_output_dir:
        # provenance copy next to the results this notebook writes
        base = globals_.get(base_dir_key)
        subdir = next((globals_[k] for k in output_dir_keys if globals_.get(k)), "")
        if base:
            try:
                target = Path(base) / subdir
                target.mkdir(parents=True, exist_ok=True)
                (target / OUTPUT_PROVENANCE_NAME).write_text(payload)
            except OSError as e:
                print(f"[recipe_params] could not write next to the output directory ({e})")

    if verbose:
        print(f"[recipe_params] wrote parameters for {record['notebook'] or '<unknown notebook>'} -> {out_dir}")
    return record
