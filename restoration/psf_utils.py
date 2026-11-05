"""PSF workflow layer: bead-based PSF extraction, QC and distillation.

This module holds the *application-specific* bead pipeline. The generic
functions it builds on live in CalmUtils (and their tests in
CalmUtils/test_restoration.py):

  calmutils.restoration.subpixel_shift   : band-limited subvoxel FFT shift
  calmutils.restoration.rl_deconvolve    : custom Richardson-Lucy core
                                           (band limit in cycles/voxel,
                                           edge="reflect" padding, damp knob)
  calmutils.restoration.resample_psf     : band-limited PSF resampling to a new
                                           voxel size, centroid on center voxel
  calmutils.restoration.psf_metrics      : profile-based PSF quantification
  calmutils.restoration.profile_metrics / axis_profile : 1D profile metrics
  calmutils.restoration.encircled_energy / radius_at_fraction
  calmutils.morphology.bead_kernel       : supersampled solid-sphere kernel
  calmutils.localization.util.nearest_neighbor_distances : point-set NN distances

Functions here (workflow layer)
-------------------------------
align_bead_stack : cut + subvoxel-align bead crops -> raw stack (pure geometry)
average_stack    : background + normalize + weighted mean of a stack -> PSF
average_psf      : one-call wrapper (align_bead_stack + average_stack)
filter_close_beads : keep-mask rejecting beads with a too-close neighbour
per_bead_metrics : per-bead QC metrics from a raw aligned stack
filter_mad_outliers : median/MAD outlier keep-mask (1.4826-scaled MAD)
distill_psf      : PSF distillation = rl_deconvolve with the bead kernel
                   (supplied kernel array or model sphere from bead_diameter+psz),
                   unit-sum output

Typical flow:
    stack, used = align_bead_stack(img, locs, psf_size)
    m = per_bead_metrics(stack, used, psz=psz)
    keep = used & filter_close_beads(locs, psz=psz, min_distance=...) \
                & filter_mad_outliers(m["fwhm_px"].mean(axis=1), n_mad=4, side="high")
    psf = average_stack(stack, keep)
    psf_dist = distill_psf(psf, bead_diameter=170.0, psz=psz, num_iter=25)
    report = psf_metrics(psf_dist, psz)   # from calmutils.restoration

Self-test (workflow integration, synthetic data with known ground truth):
`python psf_utils.py`.

TODO
----
- Depth stratification: per-bead FWHM vs. z plots; group-average PSFs by depth
  if a trend exists (refractive-index mismatch) - belongs in the user-facing
  aberration-analysis notebook.
- Stack-wide background estimate from bead-free regions.
"""

import warnings

import numpy as np

from calmutils.restoration import subpixel_shift, rl_deconvolve, resample_psf
from calmutils.restoration.deconvolution import _convolve_same
from calmutils.restoration.psf_metrics import axis_profile, profile_metrics
from calmutils.morphology import bead_kernel
from calmutils.localization.util import nearest_neighbor_distances

__all__ = [
    "align_bead_stack",
    "average_stack",
    "average_psf",
    "filter_close_beads",
    "per_bead_metrics",
    "filter_mad_outliers",
    "distill_psf",
]


# ---------------------------------------------------------------------------
# averaging of bead crops (align = pure geometry, average = statistics)
# ---------------------------------------------------------------------------

def _apply_background(crop, background):
    """Subtract a per-crop background: "shell" (NaN-aware boundary-shell median),
    a float, or None (no subtraction)."""
    if background is None:
        return crop
    if background == "shell":
        mask = np.ones(crop.shape, dtype=bool)
        mask[tuple(slice(2, s - 2) for s in crop.shape)] = False
        if not mask.any():
            mask[:] = True
        vals = crop[mask]
        vals = vals[np.isfinite(vals)]
        bg = np.median(vals) if vals.size else 0.0
        return crop - bg
    try:
        return crop - float(background)
    except (TypeError, ValueError):
        raise ValueError(f"unknown background: {background!r}") from None


def align_bead_stack(img, locations, psf_size, *, edge="skip"):
    """Cut and subvoxel-align bead crops -> raw stack. Pure geometry: no
    background subtraction, no normalization.

    Parameters
    ----------
    img : ndarray
        Raw image stack, ``(Z, Y, X)`` (or ``(Y, X)`` for 2D data).
    locations : (N, ndim) array of float
        Subvoxel bead positions in array-index coordinates, same axis order as
        ``img`` (e.g. ``(z, y, x)``). Pass localization-fit results, NOT rounded
        detections - subvoxel positions are the whole point of the alignment.
    psf_size : sequence of int
        Crop size in voxels, one odd value per axis (e.g. ``(13, 29, 29)``).
        Odd sizes guarantee a well-defined center voxel.
    edge : "skip" | "partial"
        "skip": beads whose full box does not fit into the image are marked
        unused. "partial": clipped beads are kept; the clipped crop is aligned
        and placed into the matching sub-region of the full-size stack row
        (NaN elsewhere). The FFT alignment wraps at the clipped edge, so
        partial beads can have small edge artifacts.

    Returns
    -------
    stack : (N, *psf_size) ndarray
        Raw aligned crops (background NOT subtracted), NaN outside the valid
        region. Brightness metrics (peak / sum) can be taken directly from it.
    used : (N,) bool ndarray
        True where the bead was usable (full box, or clipped in with
        edge="partial").

    Notes
    -----
    Alignment: each crop is centered on ``round(loc)``; the residual
    fractional offset ``off = loc - round(loc)`` is removed with a Fourier
    subpixel shift of ``-off`` (sign matters - the opposite sign doubles the
    jitter!).
    """
    img = np.asarray(img, dtype=float)
    locations = np.atleast_2d(np.asarray(locations, dtype=float))
    if locations.shape[1] != img.ndim:
        raise ValueError(f"locations must have {img.ndim} columns to match img dims")
    psf_size = tuple(int(s) for s in psf_size)
    if len(psf_size) != img.ndim:
        raise ValueError(f"psf_size must have {img.ndim} entries to match img dims")
    if any(s % 2 == 0 for s in psf_size):
        raise ValueError("psf_size must be odd per axis (well-defined center voxel)")
    if edge not in ("skip", "partial"):
        raise ValueError(f"unknown edge policy: {edge!r}")

    n_beads = len(locations)
    half = np.array(psf_size) // 2
    shape = np.array(img.shape)

    stack = np.full((n_beads,) + psf_size, np.nan)
    used = np.zeros(n_beads, dtype=bool)

    for i, loc in enumerate(locations):
        center = np.round(loc).astype(int)
        lo = center - half
        hi = center + half + 1  # exclusive

        clipped = np.any(lo < 0) or np.any(hi > shape)
        if clipped and edge == "skip":
            continue

        crop_sl = tuple(slice(max(l, 0), min(h, s))
                        for l, h, s in zip(lo, hi, shape))
        crop = img[crop_sl]
        if crop.size == 0:
            continue

        # subvoxel alignment: bead is at (loc - round(loc)) relative to the crop
        # center; shift content by the negative of that offset to center it.
        crop = subpixel_shift(crop, np.round(loc) - loc)

        # place into the full-size stack row at the position matching the
        # (possibly clipped) crop
        c_lo = np.maximum(lo, 0) - lo
        c_hi = np.minimum(hi, shape) - lo
        canvas_sl = tuple(slice(int(a), int(b)) for a, b in zip(c_lo, c_hi))
        stack[(i,) + canvas_sl] = crop
        used[i] = True

    return stack, used


def average_stack(stack, used=None, *, background="shell", normalize="sum",
                  weights=None):
    """Weighted average of an aligned bead stack -> PSF.

    Per crop: background subtraction (``background``), then per-bead
    normalization; the PSF is the weighted mean of the normalized crops.
    Re-averaging with a different ``used`` mask / ``weights`` / ``background``
    / ``normalize`` is cheap - no re-alignment needed (that's the point of
    splitting this from :func:`align_bead_stack`).

    Parameters
    ----------
    stack : (N, *psf_size) ndarray
        Raw aligned crops from :func:`align_bead_stack` (NaN outside the valid
        region of partial beads; NaN counts as 0 in sums).
    used : (N,) bool ndarray, optional
        Which beads to include (e.g. the align mask AND-ed with QC masks).
        Default: rows containing any finite value.
    background : "shell" | float | None
        Per-crop background subtracted before normalization. "shell" =
        NaN-aware median of the crop's boundary shell; float = user-supplied
        estimate; None = no subtraction (input already corrected).
    normalize : "sum" | "peak" | None
        Per-bead normalization after background subtraction. "sum" (default):
        unit-sum crops -> every bead contributes equally. "peak": unit peak
        height. None: raw intensities -> brightness-weighted mean.
    weights : (N,) array of float, optional
        Additional per-bead weights (multiplied with the normalized crop).

    Returns
    -------
    psf : ndarray of shape ``stack.shape[1:]``, unit sum for
    normalize="sum"/"peak" (weighted mean of raw crops for None).

    Notes
    -----
    Beads whose crop sums to <= 0 after background subtraction are skipped
    ("background ate the bead"); if none remain, ValueError is raised.
    """
    stack = np.asarray(stack, dtype=float)
    n = stack.shape[0]
    if used is None:
        used = np.isfinite(stack).any(axis=tuple(range(1, stack.ndim)))
    used = np.asarray(used, dtype=bool)
    if len(used) != n:
        raise ValueError("used must have one entry per stack row")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != n:
        raise ValueError("weights must have one entry per stack row")
    if normalize not in ("sum", "peak", None):
        raise ValueError(f"unknown normalize: {normalize!r}")

    psf = np.zeros(stack.shape[1:])
    w_used = 0.0
    for i in range(n):
        if not used[i]:
            continue
        crop = _apply_background(stack[i], background)
        if not np.isfinite(crop).any():
            continue
        if normalize == "sum":
            s = np.nansum(crop)
            if s <= 0:
                continue
            crop = crop / s
        elif normalize == "peak":
            m = np.nanmax(crop)
            if m <= 0:
                continue
            crop = crop / m
        psf += np.nan_to_num(crop, nan=0.0) * w[i]
        w_used += w[i]

    if w_used == 0:
        raise ValueError("no usable beads (all skipped or non-positive)")
    return psf / w_used


def average_psf(img, locations, psf_size, *, background="shell", normalize="sum",
                weights=None, edge="skip", return_stack=False):
    """Average subvoxel-aligned PSF crops around localized beads.

    Convenience wrapper: :func:`align_bead_stack` (crop + subvoxel shift, pure
    geometry) followed by :func:`average_stack` (background subtraction,
    per-bead normalization, weighted mean). Use the two steps directly when you
    need the aligned stack for per-bead QC and want to re-average with
    different masks / weights / background without re-aligning.

    Parameters
    ----------
    img, locations, psf_size, edge, return_stack
        See :func:`align_bead_stack`. ``return_stack=True`` additionally returns
        the raw aligned stack and the "used" mask.
    background, normalize, weights
        See :func:`average_stack`. Note: with normalize="sum" any residual
        background becomes a pedestal in the PSF and inflates its width; if
        psf_size is small enough that the shell sits on the PSF wings, the
        shell median over-subtracts - pass a float estimate instead. In dense
        bead fields neighbour wings can reach the shell (guard with
        :func:`filter_close_beads`).

    Returns
    -------
    psf : ndarray of shape ``psf_size``, normalized to unit sum.
    (psf, stack, used) : if ``return_stack=True``.
    """
    stack, used = align_bead_stack(img, locations, psf_size, edge=edge)
    psf = average_stack(stack, used, background=background, normalize=normalize,
                        weights=weights)
    if return_stack:
        return psf, stack, used
    return psf


# ---------------------------------------------------------------------------
# bead QC: rejection of close beads, per-bead metrics, MAD outliers
# ---------------------------------------------------------------------------

def filter_close_beads(locations, *, psz=None, min_distance=None, quantile=None):
    """Keep-mask rejecting beads whose nearest neighbour is too close.

    A bead's crop window is contaminated by the PSF wings of any neighbour
    inside (or near) it, so beads with a too-close nearest neighbour should
    be excluded from PSF averaging. The rejection is symmetric: both beads
    of a close pair are each other's nearest neighbour and both are
    rejected. Do not "keep the brighter one of the pair" - that re-introduces
    selection bias into the resolution statistics.

    Parameters
    ----------
    locations : (N, ndim) array of float
        Bead positions (array-index coordinates).
    psz : sequence of float, optional
        Voxel size per axis; thresholds are then in physical units, else in
        voxels (distances via
        :func:`calmutils.localization.util.nearest_neighbor_distances`).
    min_distance : float, optional
        Absolute threshold: reject NN distance < ``min_distance``. Comparable
        across files and acquisition conditions - prefer this for comparative
        studies. A sensible starting point is the crop window extent
        ``max(psf_size * psz)`` plus a few PSF FWHM; check the resulting PSF
        width vs. threshold for stability.
    quantile : float in [0, 1], optional
        Reject the lowest ``quantile`` fraction of NN distances, i.e. reject
        NN distance < ``np.quantile(nn, quantile)``. Adapts to each file's
        bead density - convenient for exploration, but applies a different
        criterion per sample.

    At least one of ``min_distance`` / ``quantile`` must be given. If both
    are given, the larger threshold wins: keep = nn >= max(min_distance,
    quantile(nn, q)).

    Returns
    -------
    (N,) bool ndarray
        True = keep. Fewer than 2 points -> all kept.
    """
    if min_distance is None and quantile is None:
        raise ValueError("pass min_distance= and/or quantile=")
    if min_distance is not None and min_distance < 0:
        raise ValueError("min_distance must be >= 0")
    if quantile is not None and not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    nn = nearest_neighbor_distances(locations, psz=psz)
    thr = -np.inf
    if min_distance is not None:
        thr = max(thr, float(min_distance))
    if quantile is not None:
        thr = max(thr, float(np.quantile(nn, quantile)))
    return nn >= thr


def per_bead_metrics(stack, used=None, *, background="shell", psz=None):
    """Per-bead QC metrics from a raw aligned stack (:func:`align_bead_stack`).

    Shape metrics (per-axis fwhm / sigma / centroid offset) are computed on
    background-subtracted crops (``background``: same semantics as in
    :func:`average_stack`); brightness (``peak``, ``sum``) comes from the raw
    stack. Entries are NaN where the bead was not used.

    Returns
    -------
    dict of arrays: peak, sum : (N,); fwhm_px, sigma_px, centroid_offset_px :
    (N, ndim); plus fwhm_phys : (N, ndim) if psz is given.
    """
    stack = np.asarray(stack, dtype=float)
    n = stack.shape[0]
    ndim = stack.ndim - 1
    if used is None:
        used = np.isfinite(stack).any(axis=tuple(range(1, stack.ndim)))
    used = np.asarray(used, dtype=bool)
    if len(used) != n:
        raise ValueError("used must have one entry per stack row")
    if psz is not None:
        psz = np.atleast_1d(np.asarray(psz, dtype=float))
        if len(psz) != ndim:
            raise ValueError(f"psz must have {ndim} entries to match stack dims")

    peak = np.full(n, np.nan)
    ssum = np.full(n, np.nan)
    fwhm = np.full((n, ndim), np.nan)
    sig = np.full((n, ndim), np.nan)
    coff = np.full((n, ndim), np.nan)
    center = np.array(stack.shape[1:]) // 2

    for i in range(n):
        if not used[i]:
            continue
        raw = stack[i]
        if not np.isfinite(raw).any():
            continue
        peak[i] = np.nanmax(raw)
        ssum[i] = np.nansum(raw)
        crop = _apply_background(raw, background)
        for ax in range(ndim):
            m = profile_metrics(axis_profile(crop, ax))
            fwhm[i, ax] = m["fwhm"]
            sig[i, ax] = m["sigma"]
            if np.isfinite(m["centroid"]):
                coff[i, ax] = m["centroid"] - center[ax]

    out = {"peak": peak, "sum": ssum, "fwhm_px": fwhm, "sigma_px": sig,
           "centroid_offset_px": coff}
    if psz is not None:
        out["fwhm_phys"] = fwhm * psz
    return out


def filter_mad_outliers(values, *, n_mad=4.0, side="both"):
    """Keep-mask from median/MAD outlier rejection (True = keep).

    The MAD is scaled by 1.4826 = 1/Phi^-1(0.75), so for Gaussian data the
    threshold corresponds to ``n_mad`` sigma. For heavy-tailed data it is a
    robust scale estimate, not literally sigma.

    Parameters
    ----------
    values : (N,) array
    n_mad : float
        Reject |value - median| > n_mad * 1.4826 * MAD.
    side : "both" | "high" | "low"
        "high" rejects only values above the threshold (e.g. aggregates with
        large sigma); "low" only below (e.g. dead/dim beads); "both" either.

    Notes
    -----
    NaN entries are always rejected. If MAD == 0 the IQR (scaled by 1/1.349)
    is used as fallback; if that is also 0 (e.g. constant input), all finite
    values are kept.
    """
    v = np.asarray(values, dtype=float)
    if n_mad < 0:
        raise ValueError("n_mad must be >= 0")
    if side not in ("both", "high", "low"):
        raise ValueError(f"unknown side: {side!r}")
    finite = np.isfinite(v)
    if not finite.any():
        return finite.copy()
    vf = v[finite]
    med = float(np.median(vf))
    scale = 1.4826 * float(np.median(np.abs(vf - med)))
    if scale <= 0:
        q1, q3 = np.percentile(vf, [25.0, 75.0])
        scale = (q3 - q1) / 1.349  # IQR -> Gaussian sigma equivalent
        if scale <= 0:
            return finite.copy()
    dev = (v - med) / scale
    if side == "both":
        keep = np.abs(dev) <= n_mad
    elif side == "high":
        keep = dev <= n_mad
    else:
        keep = dev >= -n_mad
    return keep & finite


# ---------------------------------------------------------------------------
# PSF distillation (deconvolution of the bead structure)
# ---------------------------------------------------------------------------

def distill_psf(observed_psf, kernel=None, *, bead_diameter=None, psz=None,
                num_iter=25, bandlimit=None, edge="periodic", damp=None,
                eps=1e-12):
    """Distill the PSF by deconvolving the finite bead structure (RL / ML).

    Model: ``observed_psf = true_psf * bead`` (the observed bead image is the
    PSF convolved with the bead's fluorophore distribution). With the bead
    known, this is a deconvolution with known input: Richardson-Lucy (Poisson
    MLE) recovers the PSF of an infinitesimal object
    (cf. Kano et al. 1996; Hell/van der Voort 1997; Huygens "PSF Distiller").
    Thin wrapper around :func:`calmutils.restoration.rl_deconvolve` with
    PSF-specific conveniences.

    Parameters
    ----------
    observed_psf : ndarray
        Observed bead image (e.g. the averaged PSF from :func:`average_psf`),
        odd shape. Unit sum is not required (the output is normalized). Small
        negative values from background residuals are clipped.
    kernel : ndarray, optional
        The kernel to deconvolve - the bead's fluorophore distribution - as a
        model from :func:`calmutils.morphology.bead_kernel` or a measured /
        idealized array (an averaged isolated-bead crop, a hollow-shell model,
        an ellipsoid structuring element). Must be odd-shaped, center-aligned
        and non-negative (checked in rl_deconvolve). If None, it is built as
        ``bead_kernel(psz, bead_diameter)`` and ``bead_diameter`` + ``psz``
        are required.
    bead_diameter : float, optional
        Bead diameter in the same length unit as ``psz``. Use the *measured*
        size, not the catalog value; assumes uniform staining. Only used to
        build the model kernel when ``kernel`` is not given.
    psz : sequence of float, optional
        Voxel size, one value per axis of ``observed_psf``. Required to build
        the model kernel and to interpret ``bandlimit``.
    num_iter, edge, damp, eps
        Forwarded to :func:`calmutils.restoration.rl_deconvolve`. ``edge``
        defaults to "periodic", which is correct for a PSF window that decays
        at the border.
    bandlimit : sequence of float, optional
        Per-axis OTF cutoff in cycles per UNIT LENGTH (same unit as ``psz``,
        which is then required), e.g. lateral ~2*NA/lambda_em for widefield
        detection (about double that for confocal); converted to cycles per
        voxel internally. None = no explicit band limit.

    Returns
    -------
    psf : ndarray, same shape as ``observed_psf``, unit sum, non-negative.

    Notes
    -----
    - Beads much larger than the PSF (e.g. 170 nm beads with strong STED)
      make the inverse problem ill-conditioned: the sphere spectrum decays
      quickly. Use the smallest beads that give you enough signal, and
      check the result against a theoretical PSF width.
    """
    img = np.asarray(observed_psf, dtype=float)

    psz_arr = None
    if psz is not None:
        psz_arr = np.atleast_1d(np.asarray(psz, dtype=float))
        if len(psz_arr) != img.ndim:
            raise ValueError(f"psz must have {img.ndim} entries to match "
                             "observed_psf dims")

    if kernel is None:
        if bead_diameter is None or psz_arr is None:
            raise ValueError("pass either kernel= or both bead_diameter= and psz=")
        kernel = bead_kernel(psz_arr, bead_diameter)

    bl_cyv = None
    if bandlimit is not None:
        if psz_arr is None:
            raise ValueError("psz is required to interpret bandlimit "
                             "(cycles per unit length)")
        bl_cyv = np.atleast_1d(np.asarray(bandlimit, dtype=float)) * psz_arr

    u = rl_deconvolve(img, kernel, num_iter=num_iter, bandlimit=bl_cyv,
                      edge=edge, damp=damp, eps=eps)
    s = u.sum()
    if s <= 0:
        raise ValueError("deconvolution produced no positive signal")
    return u / s


# ---------------------------------------------------------------------------
# self-test: workflow integration on synthetic data with known ground truth
# (the generic functions' unit tests live in CalmUtils/test_restoration.py)
# ---------------------------------------------------------------------------

def _gaussian_volume(shape, sigma):
    grids = np.meshgrid(*[np.arange(s) - (s - 1) / 2.0 for s in shape],
                        indexing="ij")
    g = np.zeros(shape)
    for ax, s in zip(grids, sigma):
        g = g + (ax / s) ** 2
    return np.exp(-0.5 * g)


def _axis_sigma(psf, ax):
    m = psf.sum(axis=tuple(j for j in range(psf.ndim) if j != ax))
    c = len(m) // 2
    return np.sqrt((((np.arange(len(m)) - c) ** 2) * m).sum() / m.sum())


def _selftest():
    rng = np.random.default_rng(42)
    ok = True

    # ---------------- average_psf / align / average ----------------
    sigma_beads = (1.8, 2.4, 2.4)  # (z, y, x) in px
    sigma = sigma_beads
    shape = (64, 128, 128)
    img = rng.poisson(5, shape).astype(float)
    zz, yy, xx = np.meshgrid([16, 32, 48], [20, 60, 100], [20, 64, 108],
                             indexing="ij")
    locs = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(float)
    locs += rng.uniform(-0.5, 0.5, locs.shape)
    zi, yi, xi = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    for loc in locs:
        img += 2000.0 * np.exp(
            -0.5 * (((zi - loc[0]) / sigma[0]) ** 2
                    + ((yi - loc[1]) / sigma[1]) ** 2
                    + ((xi - loc[2]) / sigma[2]) ** 2))
    # one extra bead close to the y/x border (for the edge-policy test below)
    border_true = np.array([16.0, 4.0, 4.0])
    img += 2000.0 * np.exp(
        -0.5 * (((zi - border_true[0]) / sigma[0]) ** 2
                + ((yi - border_true[1]) / sigma[1]) ** 2
                + ((xi - border_true[2]) / sigma[2]) ** 2))

    psf_size = (13, 21, 21)
    psf, stack, used = average_psf(img, locs, psf_size, return_stack=True)
    assert used.all(), "expected all synthetic beads to be usable"
    center = np.array(psf_size) // 2

    print(f"[average_psf] sum = {psf.sum():.6f}  (expected 1.0)")
    ok &= abs(psf.sum() - 1.0) < 1e-6

    com = np.zeros(3)
    for ax in range(3):
        m = psf.sum(axis=tuple(j for j in range(3) if j != ax))
        com[ax] = (np.arange(len(m)) * m).sum() / m.sum()
    print(f"[average_psf] centroid = {np.round(com, 3)}  (expected {center})")
    ok &= np.allclose(com, center, atol=0.1)

    for ax, sig_true in enumerate(sigma):
        s = _axis_sigma(psf, ax)
        print(f"[average_psf] sigma[{ax}]: {s:.3f} px, true {sig_true} px "
              f"({100 * (s / sig_true - 1):+.1f}%)")
        ok &= abs(s / sig_true - 1) < 0.05

    img2d = img.max(axis=0)
    psf2d = average_psf(img2d, locs[:, 1:], (21, 21))
    s2d = _axis_sigma(psf2d, 1)
    print(f"[average_psf] 2D sigma_x: {s2d:.3f} px, true {sigma[2]} px")
    ok &= abs(s2d / sigma[2] - 1) < 0.05

    # ---------------- distill_psf (wrapper over calmutils rl_deconvolve) ----------------
    psz = (150.0, 50.0, 50.0)          # nm
    bead_diameter = 100.0              # nm -> radius_px = (0.33, 1.0, 1.0)
    sigma_true = (3.0, 2.0, 2.0)       # px
    g = _gaussian_volume((25, 21, 21), sigma_true)
    kernel = bead_kernel(psz, bead_diameter)
    bead_img = _convolve_same(g, kernel)
    r_px = bead_diameter / 2.0 / np.array(psz)
    sig_eff = [np.sqrt(s ** 2 + rr ** 2 / 5.0)
               for s, rr in zip(sigma_true, r_px)]  # ball variance r^2/5

    distilled = distill_psf(bead_img, bead_diameter=bead_diameter, psz=psz,
                            num_iter=80)
    for ax in range(3):
        s_in = _axis_sigma(bead_img, ax)
        s_out = _axis_sigma(distilled, ax)
        print(f"[distill_psf] sigma[{ax}]: convolved {s_in:.3f} "
              f"(expected ~{sig_eff[ax]:.3f}) -> distilled {s_out:.3f} px, "
              f"true {sigma_true[ax]} px")
        ok &= abs(s_out / sigma_true[ax] - 1) < 0.06
        ok &= s_out < s_in  # must have sharpened
    print(f"[distill_psf] sum = {distilled.sum():.6f}, min = {distilled.min():.2e}")
    ok &= abs(distilled.sum() - 1.0) < 1e-6
    ok &= distilled.min() >= 0.0

    # kernel-array path must be identical to the bead_diameter+psz path
    distilled_k = distill_psf(bead_img, kernel=bead_kernel(psz, bead_diameter),
                              psz=psz, num_iter=80)
    same = np.allclose(distilled, distilled_k)
    print(f"[distill_psf] kernel-array path == bead_diameter path: {same}")
    ok &= same

    # bandlimit (physical units, needs psz) still recovers the PSF
    dist_bl = distill_psf(bead_img, kernel=bead_kernel(psz, bead_diameter),
                          psz=psz, num_iter=80, bandlimit=0.45 / np.array(psz))
    s_bl = [_axis_sigma(dist_bl, ax) for ax in range(3)]
    print(f"[distill_psf] bandlimit path sigmas: {np.round(s_bl, 3)} "
          f"(true {sigma_true}), sum = {dist_bl.sum():.6f}")
    ok &= all(abs(s_bl[ax] / sigma_true[ax] - 1) < 0.10 for ax in range(3))
    ok &= abs(dist_bl.sum() - 1.0) < 1e-6

    # ---------------- edge cases ----------------
    # single bead
    p1 = average_psf(img, locs[:1], psf_size)
    s1 = _axis_sigma(p1, 2)
    print(f"[edge] single bead: sum = {p1.sum():.6f}, sigma_x = {s1:.3f} px "
          f"(true {sigma_beads[2]})")
    ok &= abs(p1.sum() - 1.0) < 1e-6 and abs(s1 / sigma_beads[2] - 1) < 0.05

    # bead at the image border: skipped by default, kept with edge="partial";
    # a border-only input with edge="skip" raises (no usable beads)
    border = (border_true + rng.uniform(-0.5, 0.5, 3))[None, :]
    two = np.vstack([border, locs[:1]])
    psf_skip, _, used_skip = average_psf(img, two, psf_size, return_stack=True)
    p_part, _, used_part = average_psf(img, two, psf_size, edge="partial",
                                       return_stack=True)
    try:
        average_psf(img, border, psf_size)
        skip_raises = False
    except ValueError:
        skip_raises = True
    print(f"[edge] border bead: used with edge='skip' = {used_skip[0]}, "
          f"edge='partial' = {used_part[0]}, border-only+skip raises = {skip_raises}, "
          f"sums: skip {psf_skip.sum():.6f} / partial {p_part.sum():.6f}")
    ok &= (not used_skip[0]) and used_skip[1] and used_part[0] and skip_raises
    ok &= abs(psf_skip.sum() - 1.0) < 1e-6 and abs(p_part.sum() - 1.0) < 1e-6

    # bead smaller than one voxel -> kernel degenerates to a delta ->
    # distillation is a no-op (compare against the input's own measured sigma)
    g2 = _gaussian_volume((21, 25), (2.0, 3.0))
    d2 = distill_psf(g2, bead_diameter=10.0, psz=(50.0, 50.0), num_iter=10)
    s_in2 = (_axis_sigma(g2, 0), _axis_sigma(g2, 1))
    s_noop = (_axis_sigma(d2, 0), _axis_sigma(d2, 1))
    print(f"[edge] sub-voxel bead: distill no-op sigmas {np.round(s_noop, 3)} "
          f"vs input {np.round(s_in2, 3)}")
    ok &= abs(s_noop[0] / s_in2[0] - 1) < 0.02 and abs(s_noop[1] / s_in2[1] - 1) < 0.02

    # ---------------- QC: filter_close_beads / per_bead_metrics / MAD ----------------
    pts2 = np.array([[0., 0.], [0.5, 0.], [20., 20.], [30., 30.]])
    keep = filter_close_beads(pts2, min_distance=1.0)
    keep_q = filter_close_beads(pts2, quantile=0.5)
    keep_both = filter_close_beads(pts2, min_distance=100.0, quantile=0.5)
    exp_keep = np.array([False, False, True, True])
    print(f"[qc] keep (min_distance=1): {keep}, (quantile=0.5): {keep_q}, "
          f"(both, min=100): {keep_both}")
    ok &= (keep == exp_keep).all() and (keep_q == exp_keep).all()
    ok &= not keep_both.any()  # larger threshold wins

    # composition with the align/average split: near-duplicate bead filtered out
    stack_a, used_a = align_bead_stack(img, locs, psf_size)
    locs_qc = np.vstack([locs, locs[0] + 0.3])
    keep_qc = filter_close_beads(locs_qc, min_distance=2.0)
    stack_qc, used_qc = align_bead_stack(img, locs_qc, psf_size)
    psf_qc = average_stack(stack_qc, used_qc & keep_qc)
    print(f"[qc] close pair rejected: {not keep_qc[0] and not keep_qc[-1]}, "
          f"rest kept: {bool(keep_qc[1:-1].all())}, psf sum = {psf_qc.sum():.6f}")
    ok &= (not keep_qc[0]) and (not keep_qc[-1]) and bool(keep_qc[1:-1].all())
    ok &= abs(psf_qc.sum() - 1.0) < 1e-6

    # split API equivalence + raw-stack property
    psf_split = average_stack(stack_a, used_a)
    psf_wrap, stack_w, used_w = average_psf(img, locs, psf_size, return_stack=True)
    rel = np.abs(psf_split - psf_wrap).max() / psf_wrap.max()
    print(f"[split] used identical: {bool((used_a == used_w).all())}, "
          f"stack identical: {bool(np.allclose(stack_a, stack_w, equal_nan=True))}, "
          f"psf max rel diff vs wrapper: {rel:.2e}")
    ok &= bool((used_a == used_w).all())
    ok &= bool(np.allclose(stack_a, stack_w, equal_nan=True)) and rel < 1e-3
    i0 = int(np.argmax(used_a))
    c0 = np.round(locs[i0]).astype(int)
    half = np.array(psf_size) // 2
    raw_crop = img[tuple(slice(c - h, c + h + 1) for c, h in zip(c0, half))]
    rel_sum = abs(np.nansum(stack_a[i0]) - raw_crop.sum()) / raw_crop.sum()
    ok &= rel_sum < 1e-6  # stack rows are raw: shift preserves the sum
    psf_mean = average_stack(stack_a, used_a, background=None, normalize=None)
    ok &= np.allclose(psf_mean, stack_a[used_a].mean(axis=0), rtol=1e-9, atol=1e-12)
    print(f"[split] raw stack row sum rel diff {rel_sum:.2e}; "
          f"normalize=None == raw mean: True")

    # per-bead metrics on the aligned stack (border bead -> NaN row)
    mb = per_bead_metrics(stack_a, used_a, psz=(150.0, 50.0, 50.0))
    fw_beads = np.array([2 * np.sqrt(2 * np.log(2)) * s for s in sigma_beads])
    med_fwhm = np.nanmedian(mb["fwhm_px"], axis=0)
    stack_b, used_b = align_bead_stack(img, two, psf_size)
    mb_b = per_bead_metrics(stack_b, used_b)
    print(f"[qc] per-bead median fwhm_px {np.round(med_fwhm, 3)} vs "
          f"{np.round(fw_beads, 3)}; peaks {np.nanmin(mb['peak']):.0f}.."
          f"{np.nanmax(mb['peak']):.0f}; border row NaN: "
          f"{bool(np.isnan(mb_b['peak'][0]))}")
    ok &= np.allclose(med_fwhm, fw_beads, rtol=0.08)
    ok &= mb["fwhm_px"].shape == (len(locs), 3) and mb["fwhm_phys"].shape == (len(locs), 3)
    ok &= bool(np.isnan(mb_b["peak"][0])) and bool(np.isfinite(mb_b["peak"][1]))

    # MAD outlier filter
    rng2 = np.random.default_rng(7)
    vals = rng2.normal(10, 1, 200)
    vals[0] = 60.0
    vals[1] = -40.0
    vals[2] = np.nan
    keep_m = filter_mad_outliers(vals, n_mad=4)
    keep_h = filter_mad_outliers(vals, n_mad=4, side="high")
    keep_l = filter_mad_outliers(vals, n_mad=4, side="low")
    keep_const = filter_mad_outliers(np.ones(10), n_mad=4)
    print(f"[qc] MAD: both rejects high+low+NaN = "
          f"{not keep_m[0] and not keep_m[1] and not keep_m[2] and bool(keep_m[3:].all())}, "
          f"side='high' keeps low = {bool(keep_h[1])}, side='low' keeps high = "
          f"{bool(keep_l[0])}, constant all kept = {bool(keep_const.all())}")
    ok &= (not keep_m[0]) and (not keep_m[1]) and (not keep_m[2]) and bool(keep_m[3:].all())
    ok &= (not keep_h[0]) and bool(keep_h[1]) and (not keep_h[2])
    ok &= bool(keep_l[0]) and (not keep_l[1]) and (not keep_l[2])
    ok &= bool(keep_const.all())

    # ---------------- end-to-end: average -> distill -> resample ----------------
    psz_nm = np.array([150.0, 50.0, 50.0])
    bead_nm = 100.0
    img_beads = _convolve_same(img, bead_kernel(psz_nm, bead_nm))
    psf_b = average_psf(img_beads, locs, psf_size)
    dist = distill_psf(psf_b, bead_diameter=bead_nm, psz=psz_nm, num_iter=80)
    for ax in range(3):
        s = _axis_sigma(dist, ax)
        print(f"[round-trip] distilled sigma[{ax}]: {s:.3f} px, "
              f"true {sigma_beads[ax]} px")
        ok &= abs(s / sigma_beads[ax] - 1) < 0.08
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        res, res_psz = resample_psf(dist, psz_nm, psz_nm / 2.0)
    edge_warned = any("window edge" in str(w.message) for w in wlist)
    print(f"[round-trip] resampled x2 -> shape {res.shape}, edge warning: {edge_warned}")
    for ax in range(3):
        s_nm = _axis_sigma(res, ax) * res_psz[ax]
        true_nm = sigma_beads[ax] * psz_nm[ax]
        print(f"[round-trip] resampled x2 sigma[{ax}]: {s_nm:.1f} nm, "
              f"true {true_nm:.1f} nm")
        ok &= abs(s_nm / true_nm - 1) < 0.08

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()
