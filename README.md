# image-analyis-recipes
"recipe book" of reusable code for various image analysis tasks

## Requirements

some recipes require our own CalmUtils (https://github.com/CALM-LMU/CalmUtils) library, install it via

```
pip install -r requirements.txt
```

or directly:

```
pip install calmutils
```

## Parameter collection / run provenance

Every parameterised recipe keeps its run parameters in a cell tagged `parameters`. Two helpers work on those cells:

**1. Runtime capture (`recipe_params.py`)** - recipes that have logging set up carry an auto-managed cell (tagged `parameter-capture`) *directly after* the `parameters` cell. It dumps the values that are in effect to `parameter_runs/<notebook>.json` (latest run) and appends a line to `parameter_runs/runs.jsonl` (append-only log).

Because papermill injects its override cell between the `parameters` cell and the capture cell, the dumped values are the ones that were *actually* used - and when you run a recipe by hand the same cell simply records the values currently in the file. Each record also lists `changed`, i.e. what deviated from the defaults in the notebook, plus git commit, timestamp, cwd and runner.

### Setting up parameter logging (one notebook, a subtree, or all)

The managed cells are inserted/refreshed by `scripts/inject_parameter_logging.py`. With no arguments it covers every notebook below the repo root that has a cell tagged `parameters`; positional arguments restrict it to directories and/or individual notebooks, and `--check`/`--remove` take the same arguments:

```
python scripts/inject_parameter_logging.py                              # all notebooks in the repo
python scripts/inject_parameter_logging.py resave/resave_nd2_as_tiff.ipynb   # just this recipe
python scripts/inject_parameter_logging.py measurement/ spot-detection/      # these subtrees (recursive)
python scripts/inject_parameter_logging.py --check                        # report drift (non-zero exit, for CI)
python scripts/inject_parameter_logging.py --remove                       # drop them again
```

Cells are identified by cell metadata (`recipe_book: {"managed": "parameter-logging"}`), not by their text, so re-running is idempotent: it reports `unchanged` when the cell already matches, `updated` when the template changed, and never touches unmanaged cells. `--check` prints `missing`/`outdated` without writing; `--remove` deletes only managed cells (`{'removed': N}`).

No capture cells are committed right now - inject a single notebook, run it, and check the record (`recipe_parameters.json` in `in_path` + `out_subfolder`, plus `parameter_runs/runs.jsonl`) before rolling it out with the repo-wide command.

A recipe can also log without any injected cell: put `recipe_params.capture(globals())` in a cell yourself (needs the repo root on `sys.path`, which is what the managed cell does for recipes living in subdirectories).

`scripts/collect_parameters.py` uses the same positional `roots` argument, plus `--no-recursive` to scan only the direct children of a directory (e.g. `python scripts/collect_parameters.py -f md measurement/`).

**2. Static scan (`scripts/collect_parameters.py`)** - parses the `parameters` cells with `ast`, nothing is executed:

```
python scripts/collect_parameters.py                          # JSON to stdout
python scripts/collect_parameters.py -f yaml -o recipe_parameters.yaml   # needs ruamel.yaml (optional)
python scripts/collect_parameters.py -f md > RECIPES.md       # overview table of all recipes
python scripts/collect_parameters.py --include-injected measurement/   # pick up papermill-injected values
```

Non-literal values (e.g. `visualization_colors = DEFAULT_COLOR_NAMES`) are reported as their source text and flagged `unresolved` in the JSON output.

`run-parameterized-nb.ipynb` sets `RECIPE_NOTEBOOK`/`RECIPE_RUNNER` (inherited by the papermill kernel) and has a "What was run?" cell that summarises the logged runs. Overridable via `RECIPE_REPO_ROOT` and `RECIPE_PARAM_OUT_DIR`. By default each run also writes `recipe_parameters.json` into the directory the recipe puts its results in (`in_path` + `out_subdirectory` &co), so the dataset keeps a copy of the parameters it was processed with (`capture(..., also_to_output_dir=False)` turns that off).

## Running pipelines (`recipe_runner.py`)

`scripts/run_pipeline.py` runs an ordered list of recipe notebooks (a pipeline) over a work list of datasets, via papermill. Both are defined in a YAML manifest:

```yaml
# pipeline.yaml
steps:
  - notebook: resave/resave_nd2_as_tiff.ipynb
    params: {out_subfolder: tif}
    done_if_exists: "{in_path}/{out_subfolder}"     # skip if output exists

  - notebook: spot-detection/RS-FISH_spot_detection.ipynb
    params: {channels: [0, 1], tif_subfolder: tif}
    done_if_exists: "{in_path}/detections/merge.csv"

runs:                                              # work list (inline …)
  - {in_path: /data/.../GS550}
  - {in_path: /data/.../GS551, channels: [1, 2]}   # per-run override
```

```
python scripts/run_pipeline.py pipeline.yaml                  # run everything
python scripts/run_pipeline.py pipeline.yaml --dry-run        # show what would run
python scripts/run_pipeline.py pipeline.yaml --only-step 2    # run just step 2
python scripts/run_pipeline.py pipeline.yaml --run-filter GS55 # only datasets whose in_path contains "GS55"
```

Parameter resolution (later wins): notebook defaults < `step.params` < run entry. `done_if_exists` is a path template formatted with the effective parameters, so it respects per-run overrides. The loop is sample-major: each dataset goes through the full pipeline before the next starts; if a step fails, the remaining steps for that dataset are skipped.

The work list can also live in a separate samples file (a YAML list of run dicts), so the same pipeline can be reused with different data:

```yaml
# pipeline.yaml
steps: …
samples: samples_jurkat.yaml     # path relative to the pipeline file (or a list of paths)
```

```
python scripts/run_pipeline.py pipeline.yaml --samples new_batch.yaml   # CLI override
python scripts/run_pipeline.py pipeline.yaml --samples a.yaml --samples b.yaml   # concatenate
```

A single experiment can be processed directly, without any samples file:

```
python scripts/run_pipeline.py pipeline.yaml --in-path /data/.../GS550
python scripts/run_pipeline.py pipeline.yaml --in-path /data/.../GS550 --param 'channels=[1, 2]'
```

`--in-path DIR` bypasses the manifest's work list (`runs:`/`samples:`/`--samples`) and processes just that one directory. `--param KEY=VALUE` (repeatable) adds run-level parameters, parsed as YAML (`1` → int, `[1, 2]` → list, `true` → bool, otherwise string). Resolution is unchanged — notebook defaults < `step.params` < run entry — so these act as per-run overrides.

See `recipe_runner.py` for the full format documentation.
