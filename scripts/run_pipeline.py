#!/usr/bin/env python3
"""Run a recipe pipeline from a YAML manifest.

Usage:
    python scripts/run_pipeline.py manifest.yaml
    python scripts/run_pipeline.py manifest.yaml --dry-run
    python scripts/run_pipeline.py manifest.yaml --samples samples.yaml
    python scripts/run_pipeline.py manifest.yaml --samples a.yaml --samples b.yaml
    python scripts/run_pipeline.py manifest.yaml --in-path /data/GS550
    python scripts/run_pipeline.py manifest.yaml --in-path /data/GS550 --param 'channels=[1, 2]'
    python scripts/run_pipeline.py manifest.yaml --only-step 2
    python scripts/run_pipeline.py manifest.yaml --run-filter GS55
    python scripts/run_pipeline.py manifest.yaml --stop-on-error

The manifest defines an ordered list of steps (notebooks + default params)
and a work list of runs (parameter dicts).  A single run can be given
directly on the command line with --in-path (plus optional --param
overrides), bypassing the manifest's work list — no samples YAML needed.
See ``recipe_runner.py`` for the full format documentation and examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recipe_runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("manifest", help="Path to the pipeline manifest YAML")
    runs_group = ap.add_mutually_exclusive_group()
    runs_group.add_argument(
        "--samples",
        action="append",
        default=None,
        metavar="FILE",
        help="Samples (work list) YAML file. Repeatable — multiple files"
        " are concatenated. Overrides 'samples:' and 'runs:' in the manifest.",
    )
    runs_group.add_argument(
        "--in-path",
        default=None,
        metavar="DIR",
        help="Process a single run with this input directory, bypassing the"
        " work list in the manifest (no samples YAML needed).",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Run-level parameter for --in-path, in YAML syntax"
        " (e.g. --param 'channels=[1, 2]'). Repeatable.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing notebooks",
    )
    ap.add_argument(
        "--only-step",
        type=int,
        default=None,
        metavar="N",
        help="Run only step N (1-based)",
    )
    ap.add_argument(
        "--run-filter",
        default=None,
        metavar="PATTERN",
        help="Only process runs whose in_path contains PATTERN",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the entire pipeline on the first failure",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-step progress output (summary still printed)",
    )
    args = ap.parse_args()

    runs_override = None
    if args.in_path is not None:
        run = {"in_path": args.in_path}
        for kv in args.param:
            key, sep, value = kv.partition("=")
            if not sep or not key:
                print(f"Error: --param expects KEY=VALUE, got: {kv!r}", file=sys.stderr)
                return 2
            run[key] = yaml.safe_load(value)
        runs_override = [run]
    elif args.param:
        print("Error: --param requires --in-path", file=sys.stderr)
        return 2

    try:
        manifest = recipe_runner.load_manifest(
            args.manifest, samples_paths=args.samples, runs_override=runs_override
        )
    except (ValueError, OSError) as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        return 2

    summary = recipe_runner.run_pipeline(
        manifest,
        dry_run=args.dry_run,
        only_step=(args.only_step - 1) if args.only_step is not None else None,
        run_filter=args.run_filter,
        stop_on_error=args.stop_on_error,
        verbose=not args.quiet,
    )

    return 1 if summary["failed"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
