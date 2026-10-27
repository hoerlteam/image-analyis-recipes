"""Generic pipeline runner for the recipe-book notebooks.

Reads a YAML manifest that defines a linear pipeline (an ordered list of
steps) and a work list (runs), then executes each step for each run via
papermill.

Manifest format (YAML)::

    defaults:                    # optional, merged into every step's params
      channels: [0, 1]

    steps:                       # ordered list of recipe notebooks
      - notebook: resave/resave_nd2_as_tiff.ipynb
        params:                  # default params for this step
          out_subfolder: tif
        done_if_exists:          # optional — skip step if path(s) exist
          "{in_path}/tif"        #   template, formatted with effective params

      - notebook: spot-detection/RS-FISH_spot_detection.ipynb
        params:
          channels: [0, 1]
          tif_subfolder: tif
        done_if_exists:
          - "{in_path}/detections/merge.csv"

    runs:                        # work list — each entry is a parameter dict
      - in_path: /data/.../GS550
      - in_path: /data/.../GS551
        channels: [1, 2]         # per-run override

Alternatively, the work list can live in a separate samples file, referenced
by the ``samples`` key (path relative to the manifest file, or a list of
paths)::

    samples: samples_jurkat.yaml          # or: samples: [batch1.yaml, batch2.yaml]

A samples file is a YAML list of run dicts (or a mapping with a ``runs``
key)::

    # samples_jurkat.yaml
    - in_path: /data/.../GS550
    - in_path: /data/.../GS551
      channels: [1, 2]

Runs resolution priority: CLI ``--samples`` > manifest ``samples:`` >
inline ``runs:``.  Multiple samples files are concatenated.

Parameter resolution (later wins)::

    notebook defaults  <  manifest defaults  <  step.params  <  run entry

Loop order is sample-major: each run goes through the full pipeline before
the next run starts.  If a step fails, the remaining steps for that run are
skipped and the runner moves on to the next run (unless ``stop_on_error``).

Usage from Python::

    import recipe_runner
    manifest = recipe_runner.load_manifest("pipelines/my_pipeline.yaml")
    recipe_runner.run_pipeline(manifest, dry_run=True)

Usage from CLI::

    python scripts/run_pipeline.py pipelines/my_pipeline.yaml --dry-run
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Ensure recipe_params is importable (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe_params  # noqa: E402


# ---------------------------------------------------------------------------
# manifest loading
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One step in the pipeline: a notebook plus default parameters."""

    notebook: str
    params: dict[str, Any] = field(default_factory=dict)
    done_if_exists: str | list[str] | None = None


@dataclass
class Manifest:
    """A parsed pipeline manifest."""

    defaults: dict[str, Any]
    steps: list[Step]
    runs: list[dict[str, Any]]


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    """Load a samples (work list) YAML file.

    Accepts either a bare list of run dicts::

        - in_path: /data/.../GS550
        - in_path: /data/.../GS551

    or a mapping with a ``runs`` key::

        runs:
          - in_path: /data/.../GS550

    Returns the list of run dicts.  Raises ``ValueError`` on structural
    problems.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if isinstance(data, list):
        runs = data
    elif isinstance(data, dict):
        runs = data.get("runs") or []
    else:
        raise ValueError(
            f"{path}: samples file must be a list or a mapping with a 'runs' key"
        )

    if not runs:
        raise ValueError(f"{path}: samples file contains no runs")

    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: run {i + 1} must be a mapping")

    return runs


def load_manifest(
    path: str | Path,
    samples_paths: list[str | Path] | None = None,
) -> Manifest:
    """Load and validate a pipeline manifest YAML file.

    Runs can come from three sources, in priority order:

    1. ``samples_paths`` — explicit list of samples file paths (e.g. from
       the CLI ``--samples`` flag).  Multiple files are concatenated.
    2. ``samples`` key in the manifest — a path (or list of paths) to
       samples file(s), resolved relative to the manifest file's directory.
    3. ``runs`` key in the manifest — an inline list of run dicts.

    Raises ``ValueError`` with a descriptive message on structural problems.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    defaults = data.get("defaults") or {}
    steps_raw = data.get("steps") or []

    if not steps_raw:
        raise ValueError(f"{path}: 'steps' must list at least one step")

    steps: list[Step] = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict) or "notebook" not in s:
            raise ValueError(f"{path}: step {i + 1} must have a 'notebook' key")
        steps.append(Step(
            notebook=s["notebook"],
            params=s.get("params") or {},
            done_if_exists=s.get("done_if_exists"),
        ))

    # Resolve runs: CLI --samples > manifest 'samples:' > inline 'runs:'
    runs: list[dict[str, Any]] = []
    if samples_paths is not None:
        for sp in samples_paths:
            runs.extend(load_samples(sp))
    elif "samples" in data:
        samples_ref = data["samples"]
        refs = [samples_ref] if isinstance(samples_ref, str) else samples_ref
        for ref in refs:
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                ref_path = path.parent / ref_path
            runs.extend(load_samples(ref_path))
    elif "runs" in data:
        runs = data["runs"]
    else:
        raise ValueError(
            f"{path}: no runs specified — use inline 'runs:',"
            f" a 'samples:' file reference, or --samples"
        )

    if not runs:
        raise ValueError(f"{path}: manifest must define at least one run")

    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: run {i + 1} must be a mapping")

    return Manifest(defaults=defaults, steps=steps, runs=runs)


# ---------------------------------------------------------------------------
# parameter resolution
# ---------------------------------------------------------------------------

def resolve_params(
    manifest: Manifest, step: Step, run: dict[str, Any]
) -> dict[str, Any]:
    """Effective parameters for a (step, run) pair, passed to papermill.

    Merge order: manifest ``defaults`` < ``step.params`` < ``run``.
    Notebook defaults are the implicit fallback inside the notebook itself.
    """
    return {**manifest.defaults, **step.params, **run}


def _notebook_defaults(notebook_path: Path) -> dict[str, Any]:
    """Best-effort static extraction of a notebook's parameter-cell defaults."""
    try:
        info = recipe_params.parameters_from_notebook(
            notebook_path, include_injected=False
        )
        return {k: v["value"] for k, v in info["defaults"].items()}
    except Exception:
        return {}


def _full_params_for_template(
    manifest: Manifest,
    step: Step,
    run: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Parameters for ``done_if_exists`` template formatting.

    Includes notebook defaults so templates can reference parameters that
    are only defined in the notebook's ``parameters`` cell.
    """
    nb_path = _resolve_notebook_path(step.notebook, repo_root)
    nb_defaults = _notebook_defaults(nb_path) if nb_path.is_file() else {}
    return {**nb_defaults, **resolve_params(manifest, step, run)}


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def _resolve_notebook_path(notebook: str, repo_root: Path) -> Path:
    """Resolve a notebook path relative to the repo root (or absolute)."""
    p = Path(notebook)
    return p if p.is_absolute() else repo_root / p


def _resolve_output_path(path_str: str, repo_root: Path) -> Path:
    """Resolve an output path (absolute, or relative to repo root)."""
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


# ---------------------------------------------------------------------------
# skip check
# ---------------------------------------------------------------------------

def check_done(
    done_if_exists: str | list[str] | None,
    full_params: dict[str, Any],
    repo_root: Path,
) -> bool:
    """Return True if a step's output already exists and it can be skipped.

    ``done_if_exists`` is a path template (or list of templates) formatted
    with the full effective parameters.  When a list is given, *all* paths
    must exist for the step to be considered done.
    """
    if not done_if_exists:
        return False
    templates = [done_if_exists] if isinstance(done_if_exists, str) else done_if_exists
    paths: list[Path] = []
    for t in templates:
        try:
            path_str = t.format(**full_params)
        except KeyError as e:
            raise ValueError(
                f"done_if_exists template references unknown parameter: {e}"
            )
        paths.append(_resolve_output_path(path_str, repo_root))
    return all(p.exists() for p in paths)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def run_step(
    step: Step,
    params: dict[str, Any],
    repo_root: Path,
    dry_run: bool = False,
) -> bool:
    """Execute one step via papermill.  Returns True if the notebook ran."""
    import papermill as pm

    nb_path = _resolve_notebook_path(step.notebook, repo_root)
    if not nb_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    if dry_run:
        print(f"    [dry-run] {step.notebook}  params={params}")
        return False

    # Save/restore env vars so recipe_params.capture() can identify the run
    old_wd = os.getcwd()
    saved = {k: os.environ.get(k) for k in ("RECIPE_NOTEBOOK", "RECIPE_RUNNER")}
    try:
        os.chdir(nb_path.parent)
        os.environ["RECIPE_NOTEBOOK"] = step.notebook
        os.environ["RECIPE_RUNNER"] = "papermill"
        pm.execute_notebook(nb_path.name, "/dev/null", parameters=params)
    finally:
        os.chdir(old_wd)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    return True


# ---------------------------------------------------------------------------
# pipeline loop
# ---------------------------------------------------------------------------

def run_pipeline(
    manifest: Manifest,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
    only_step: int | None = None,
    run_filter: str | None = None,
    stop_on_error: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """Run the pipeline: for each run, execute each step in order.

    Args:
        manifest: Parsed pipeline manifest.
        repo_root: Repo root — notebooks are resolved relative to this.
            Defaults to ``recipe_params.repo_root()``.
        dry_run: Print what would run without executing notebooks.
        only_step: If set (0-based index), only run this step.
        run_filter: If set, only process runs whose ``in_path`` contains
            this substring.
        stop_on_error: Abort the entire pipeline on the first failure.
            By default, a failure skips the remaining steps for that run
            and continues with the next run.
        verbose: Print progress to stdout.

    Returns:
        Summary dict with keys ``total``, ``executed``, ``skipped``,
        ``failed``.
    """
    if repo_root is None:
        repo_root = recipe_params.repo_root()
    repo_root = Path(repo_root)

    summary = {"total": 0, "executed": 0, "skipped": 0, "failed": 0}

    for run_idx, run in enumerate(manifest.runs):
        # Run filter
        if run_filter is not None:
            in_path = str(run.get("in_path", ""))
            if run_filter not in in_path:
                continue

        label = run.get("in_path", f"run {run_idx + 1}")
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Run {run_idx + 1}/{len(manifest.runs)}: {label}")
            print(f"{'=' * 60}")

        for step_idx, step in enumerate(manifest.steps):
            if only_step is not None and step_idx != only_step:
                continue

            summary["total"] += 1
            params = resolve_params(manifest, step, run)
            full_params = _full_params_for_template(manifest, step, run, repo_root)

            # Skip check
            if check_done(step.done_if_exists, full_params, repo_root):
                if verbose:
                    print(
                        f"  Step {step_idx + 1}/{len(manifest.steps)}"
                        f"  {step.notebook}  SKIP (output exists)"
                    )
                summary["skipped"] += 1
                continue

            if verbose:
                print(
                    f"  Step {step_idx + 1}/{len(manifest.steps)}"
                    f"  {step.notebook}  running..."
                )

            try:
                run_step(step, params, repo_root, dry_run=dry_run)
                summary["executed"] += 1
                if verbose and not dry_run:
                    print("    done.")
            except Exception as e:
                summary["failed"] += 1
                print(f"    FAILED: {e}", file=sys.stderr)
                if stop_on_error:
                    print("Stopping (--stop-on-error).", file=sys.stderr)
                    return summary
                # Linear pipeline: skip remaining steps for this run
                remaining = len(manifest.steps) - step_idx - 1
                if verbose and remaining > 0:
                    print(f"    Skipping remaining {remaining} step(s) for this run.")
                break

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Summary: {summary['executed']} executed,"
            f" {summary['skipped']} skipped,"
            f" {summary['failed']} failed"
            f" (of {summary['total']} total)"
        )
        print(f"{'=' * 60}")

    return summary
