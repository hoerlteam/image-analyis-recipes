# PSF / restoration code — current-state reference

Reference for the bead-PSF measurement and deconvolution code split across
CalmUtils and image-analyis-recipes. (extracted from older exploratory notebooks in private scratchbook repo)

## Layout

| Layer | Location | Contents |
|---|---|---|
| Generic library | `CalmUtils/calmutils/restoration/` (`shift.py`, `deconvolution.py`, `resampling.py`, `psf_metrics.py`), `calmutils/morphology/structuring_elements.py` (`bead_kernel`), `calmutils/localization/util/point_distances.py` | array-level functions, no microscopy domain assumptions |
| Recipe-book helpers | repo root `recipe_utils.py` | shared recipe *policy* (not algorithms): filename grouping (`assign_groups`), filename-safe stems (`sanitize_stem`); imported by notebooks and domain modules |
| Bead workflow layer | `image-analyis-recipes/restoration/psf_utils.py` | bead pipeline glue; imports CalmUtils one-way, re-exports `assign_groups` from `recipe_utils.py` |
| User-facing notebooks | `image-analyis-recipes/restoration/*.ipynb` | `psf_extraction`, `deconvolution`, `psf_aberrations` |
| Tests | `CalmUtils/test_restoration.py` (generic, run directly), `python recipe_utils.py` (recipe helpers), `python restoration/psf_utils.py` (workflow self-test) | |

Environment: env-py314 (Python 3.14.6, numpy 2.4.6, scipy, scikit-image,
papermill). Both suites pass, clean under `-W error::DeprecationWarning`:
`TEST PASSED` and `SELFTEST PASSED`.

## Notebooks (all: one dataset folder in, results subfolder out, `parameters` cell tagged for papermill / `recipe_runner.py`)

- **`psf_extraction.ipynb`** — `in_path/<tif_subdir>/*.tif` (ImageJ tiffs;
  psz per file from metadata, or common `psz` param) → `in_path/<out_subdir>/`:
  pooled `<group>_psf_*.tif` + sidecars per group (only if all
  files of the group share one pixel size; otherwise warn + per-file only for
  that group), `detections.csv` (pooled, QC mask columns + `group`),
  `aligned_stack.npz` (incl. `psz_per_file`, `group_idx`), report figures
  `report_<group>.png` (orthogonal cuts in
  nm + marginals; `_distilled` variants when distilling), provenance.
  Per-file outputs (`<stem>_psf_{avg,distilled}.tif` + sidecars, per-file
  `detections_<stem>.csv`, `report_<stem>.png`) are gated by `save_per_file`
  (None = auto: on when `group_patterns` is None, off when grouping — the pooled
  `detections.csv` + `aligned_stack.npz` then suffice to reconstruct per-file
  results; per-file PSFs are still computed in memory for the outlier warning).
  Grouping via `group_patterns` (dict group name → filename glob or list of
  globs; a list is OR-ed within a group, use one combined glob for AND;
  first match wins, files matching no pattern are excluded with a
  warning; None → single group `ALL`, i.e. previous behavior / `ALL_*` outputs).
  Pipeline: LoG detect + `refine_point_lsq` → `align_bead_stack` per file,
  pooled → `per_bead_metrics` + `filter_close_beads` (per file) +
  `filter_mad_outliers` (per group) → `average_stack` per file and per group
  (pooled only if psz uniform within the group) → optional `distill_psf` →
  `psf_metrics` report. Runner skip:
  `done_if_exists: "{in_path}/psf/ALL_psf_avg.tif"` (with `group_patterns`, use
  e.g. `"{in_path}/psf/ch488_psf_avg.tif"`).
- **`deconvolution.ipynb`** — `in_path/<tif_subdir>/*.tif` + explicit `psf_path`
  (PSF tiff from extraction; psz from its `.json` sidecar) →
  `resample_psf` onto each image grid → `rl_deconvolve(edge="reflect", num_iter,
  bandlimit, damp)` → `<stem>_deconv.tif` + sidecar + `deconv_summary.csv` +
  `report.png`.
- **`psf_aberrations.ipynb`** (draft) — reads an extraction `psf_dir`
  (`aligned_stack.npz`, `detections.csv`, sidecar) → `local_psf` on a (y,x)
  grid × z levels if `do_depth` (`'auto'` = only if kept beads span >
  `depth_min_span_nm` in z; single coverslip layer → skipped) →
  `local_psf_metrics.csv`, `per_bead_metrics.csv`, `maps.png`, `depth.png`.
  capture called with `base_dir_key='psf_dir'`.

All three verified end-to-end with papermill on synthetic data (0 errors):
extraction pooled FWHM [639.5, 283.0, 283.0] nm vs true [635.8, 282.6, 282.6];
deconvolution blob sigma [1.96, 2.50, 2.50] → [1.51, 1.61, 1.61] px (true 1.5),
flux out/in 1.002; aberration local FWHM xy ≈ 283 nm, both depth modes.

## Function reference

Workflow layer (`restoration/psf_utils.py`):

| Function | Purpose |
|---|---|
| `align_bead_stack(img, locations, psf_size, *, edge="skip")` | pure geometry: crop + subvoxel shift → raw `stack (N, *psf_size)` + `used`; no background/normalization |
| `average_stack(stack, used=None, *, background="shell", normalize="sum", weights=None)` | per-crop background → per-bead normalize → weighted mean → PSF; cheap re-averaging with different masks/weights |
| `average_psf(...)` | one-call wrapper of the two above; `return_stack=True` also returns stack/used |
| `filter_close_beads(locations, *, psz, min_distance=None, quantile=None)` | keep-mask vs too-close nearest neighbour; symmetric pair rejection; larger threshold wins |
| `per_bead_metrics(stack, used=None, *, background="shell", psz=None)` | per-bead peak/sum (raw stack) + per-axis fwhm/sigma/centroid-offset (bg-subtracted crops) |
| `filter_mad_outliers(values, *, n_mad=4.0, side="both")` | MAD keep-mask (×1.4826); NaN rejected; MAD=0 → IQR fallback |
| `distill_psf(observed_psf, kernel=None, *, bead_diameter=None, psz=None, ...)` | PSF distillation = `rl_deconvolve` + bead kernel (array, or model sphere from `bead_diameter`+`psz`); unit-sum out |
| `local_psf(stack, used, locations, psz, center, bandwidth, ...)` → `(psf, ess)` | kernel-weighted local PSF; ess = Kish ESS noise flag |
| `assign_groups(filenames, patterns=None, *, default_group='ALL')` → `(group_names, group_idx)` | in `recipe_utils.py` (repo root; re-exported by `psf_utils.py`): filename-glob grouping of input files into named groups; `group_idx == -1` = matched no pattern (excluded) |

CalmUtils:

| Function | Location |
|---|---|
| `subpixel_shift(img, shift)` — `out[x]=img[x−shift]`; pass `−off` to center | `restoration/shift.py` |
| `rl_deconvolve(observed, kernel, *, num_iter, bandlimit (cycles/voxel), edge, damp, eps)` | `restoration/deconvolution.py` |
| `resample_psf(psf, psz_from, psz_to, *, out_size)` → `(psf, psz_eff)` | `restoration/resampling.py` |
| `axis_profile`, `profile_metrics` (+hw/asymmetry/skew), `psf_metrics` (+asymmetry/astigmatism/per-side sidelobe/ring radius), `encircled_energy`, `radius_at_fraction` — all nonparametric | `restoration/psf_metrics.py` |
| `bead_kernel(psz, bead_diameter, *, supersample=4)` | `morphology/structuring_elements.py` |
| `nearest_neighbor_distances`, `kernel_weights`, `effective_sample_size` | `localization/util/point_distances.py` |

Typical QC flow:
`stack, used = align_bead_stack(img, locs, psf_size)` →
`m = per_bead_metrics(stack, used, psz=psz)` →
`keep = used & filter_close_beads(locs, psz=psz, min_distance=...) & filter_mad_outliers(m['fwhm_px'].mean(1), n_mad=4, side='high')` →
`psf = average_stack(stack, keep)` → `psf_metrics(psf, psz)`.

## Design decisions

- **Subpixel shift sign**: positive shift moves content to higher indices; to
  center a bead at fractional offset `off`, pass `−off`. The opposite sign
  doubles the jitter. (FFT shift uses centered `fftfreq`.)
- **align vs average split**: `align_bead_stack` is pure geometry (raw stack);
  background/normalization/weights live in `average_stack` (swappable, cheap
  re-averaging). Shell median is NaN-aware; caveat: small `psf_size` puts the
  shell on PSF wings → pass a float estimate.
- **`distill_psf` takes a kernel array** (model sphere, measured bead crop,
  hollow shell, selem); kernel must be odd/centered/non-negative (guards warn;
  recenter off-center kernels with `subpixel_shift`). `bandlimit` physical in
  the wrapper, cycles/voxel in the core (`nu * psz`).
- **Custom RL, not a skimage wrapper**: per-iteration OTF band-limit mask and
  integrated `edge="reflect"` padding don't exist in skimage; skimage is used
  only as a test oracle (agreement 0.000 rel diff). **No TV regularization**
  (biases smooth PSFs, breaks intensity linearity); band limit + `num_iter` +
  `damp` are the regularizers. Core output is flux-preserving; only the
  distill wrapper renormalizes to unit sum. `edge="reflect"` pads by kernel
  half-width per axis (before, after) pairs, then crops.
- **Sidelobe ratio** = max beyond the *first local minimum* after the central
  peak of through-center cuts (3-sample smoothed); a clean Gaussian has none →
  ~0. Ring radius = distance of that maximum.
- **MAD scale 1.4826** = 1/Φ⁻¹(0.75): Gaussian σ calibration; robust scale
  otherwise.
- **NN rejection is symmetric** (both beads of a close pair); absolute
  `min_distance` preferred for cross-condition comparisons (heuristic:
  crop extent `max(psf_size)·psz`); `quantile` mode for exploration.
- **No stem auto-matching** between bead PSFs and sample images — `psf_path`
  is explicit in the deconvolution notebook.
- **File grouping** (`group_patterns`, via `assign_groups` in `recipe_utils.py`): None = single `ALL` group, so runner
  `done_if_exists` on `ALL_psf_avg.tif` and all previous outputs are unchanged.
  With patterns, everything that pools (group average, MAD filter, psz-uniformity
  check, per-file FWHM outlier warning) is per group — different channels/depths
  can have genuinely different FWHM, a global MAD/median would reject the wrong
  beads. Per-file PSFs/reports are always written. Unmatched files are excluded
  with a warning, never silently pooled; multi-match → first pattern wins +
  warning; group names are sanitized to safe filename stems.
- **Depth analysis auto-gated** on z-span (single-layer samples skip it).
- **Metrics are nonparametric** (profile FWHM/sigma, EE, …) — no Gaussian
  fits for STED PSFs; Gaussian fitting for localization stays in
  `calmutils.localization` (`refine_point_lsq`).

## Validation

- `CalmUtils/test_restoration.py`: shift sign, conv alignment, bead kernel,
  RL vs skimage + flux + bandlimit + damp + reflect (2D corner and 3D
  non-cubic kernel) + kernel guards, resample up/down/×1.5/round-trip +
  warnings, profile/psf metrics (Gaussian, donut, astigmatism), EE vs analytic
  chi-square, NN distances, kernel weights/ESS → `TEST PASSED`.
- `restoration/psf_utils.py` self-test: averaging (3D/2D, centroid, sigma),
  edge cases (single/border bead, sub-voxel bead no-op), QC masks, split-API
  equivalence, per-bead metrics, MAD, distill wrapper, local_psf, end-to-end
  average→distill→resample round trip → `SELFTEST PASSED`.
- Notebooks: papermill-executed on synthetic data, numbers above. Grouping
  verified on a synthetic 2-channel dataset (ch0/ch1 with different true FWHM
  plus one unmatched file): per-group `<group>_psf_*.tif`/reports/sidecars
  (`group` + `files` in sidecar), `group` column in `detections.csv`,
  `group_idx` in `aligned_stack.npz`, unmatched file excluded with warning;
  `group_patterns=None` reproduces the `ALL_*` outputs; mixed pixel size within
  a group → warning, per-file PSFs only for that group while other groups still
  pool.

## Known limitations (also in code docstrings)

- `distill_psf`: ill-conditioned when bead ≳ PSF (e.g. 170 nm beads with strong
  STED); use smallest beads with enough signal; assumes uniform staining and
  a *measured* bead size; STED PSFs are dye/power-specific.
- `resample_psf`: odd shapes only; `psz_eff` ≈ requested within rounding
  (warns >5%); periodic — PSF must decay at window edges (warned).
- `align_bead_stack` with `edge="partial"`: FFT alignment wraps at clipped
  edges; default `skip`.
- `average_stack`: residual background becomes a pedestal under
  `normalize="sum"` and inflates width.

## Open items

1. **First real-data runs** — tune/validate defaults: `blob_threshold_rel`,
   `min_nn_distance_nm` (heuristic `max(psf_size)·psz`), `bandwidth_nm` /
   `grid_step_nm`, `num_iter`; watch memory of `aligned_stack.npz`
   (`save_aligned_stack` toggle).
2. Stack-wide background estimate from bead-free regions (standalone helper;
   currently per-crop shell median in `average_stack`).
3. Optional: iterative bead re-registration against the current average
   PSF (re-align, re-average).
