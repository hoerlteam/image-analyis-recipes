"""Shared recipe-book helpers: file grouping / naming policy.

Layering of this repository:

- **CalmUtils** (PyPI package): application-independent, array-level
  algorithms with no microscopy/recipe assumptions (``subpixel_shift``,
  ``rl_deconvolve``, ``psf_metrics``, ``bead_kernel``, ...).
- **recipe_utils** (this module, repo root): recipe-book *policy* - how our
  recipes select, group and name input/output files. Not published, no
  algorithmics. Domain workflow modules (``restoration/psf_utils.py``,
  ``resave/nd2_resaving.py``, ...) import from here.
- **domain modules / notebooks**: per-domain workflow glue and user-facing
  recipes.

Notebooks can simply ``import recipe_utils`` after the parameter-capture cell
(has put the repo root on ``sys.path``); standalone modules use the same
cwd-and-parents search, e.g.::

    import sys
    from pathlib import Path
    for _p in [Path.cwd(), *Path.cwd().parents]:
        if (_p / 'recipe_utils.py').is_file():
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            break

Self-test: ``python recipe_utils.py``.

Currently provided:
    sanitize_stem : make an arbitrary label safe as a filename stem
    assign_groups : assign files to named groups by filename glob patterns
                    (None -> a single default group); used for per-group
                    pooling in multi-file recipes (e.g. psf_extraction)
"""

import re
import warnings
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np

__all__ = ["sanitize_stem", "assign_groups"]


def sanitize_stem(name):
    """Make an arbitrary label safe as a filename stem (PSF tiffs, reports, ...).

    Replaces every run of characters outside ``[A-Za-z0-9._-]`` with ``_`` and
    strips leading/trailing ``.``/``_``; raises if nothing usable remains.
    """
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    if not s:
        raise ValueError(f"name {name!r} has no usable characters for a filename")
    return s


def assign_groups(filenames, patterns=None, *, default_group="ALL"):
    """Assign input files to named groups via filename glob patterns.

    Used to pool multi-file recipes per optical condition (channel, depth,
    magnification, ...) instead of over all files at once.

    Parameters
    ----------
    filenames : sequence of str / Path
        Input files; only the file *name* is matched against the patterns.
    patterns : dict (or list of (name, patterns) pairs), optional
        Mapping ``group name -> glob pattern`` or list of glob patterns
        (``fnmatch`` syntax, case-sensitive, e.g. ``'*_ch0_*'``). If None or
        empty, all files go into a single group named ``default_group`` - the
        previous single-pool behavior.
    default_group : str
        Group name used when ``patterns`` is None.

    Returns
    -------
    group_names : list of str
        Unique group names (sanitized via :func:`sanitize_stem`), in pattern
        order.
    group_idx : (N,) int ndarray
        Group index per file; -1 = the file matched no pattern (excluded by
        the caller).

    Notes
    -----
    - A file matching several groups is assigned to the first one (pattern
      order); a warning lists all its matches.
    - Files matching no pattern get index -1 and a warning lists them - pooling
      unmatched leftovers silently would corrupt group averages.
    - Patterns that match no file produce a warning.
    """
    names = [Path(str(f)).name for f in filenames]
    if patterns is None or len(patterns) == 0:
        return [sanitize_stem(default_group)], np.zeros(len(names), dtype=int)

    items = patterns.items() if isinstance(patterns, dict) else patterns
    group_names, pat_lists = [], []
    for raw_name, pats in items:
        gname = sanitize_stem(raw_name)
        if gname in group_names:
            raise ValueError(f"group names collide after sanitizing: "
                             f"{raw_name!r} -> {gname!r}")
        if isinstance(pats, (str, Path)):
            pats = [pats]
        group_names.append(gname)
        pat_lists.append([str(p) for p in pats])

    n = len(names)
    group_idx = np.full(n, -1, dtype=int)
    match_count = np.zeros(n, dtype=int)
    all_matches = [[] for _ in range(n)]
    for gi, pats in enumerate(pat_lists):
        matched_any = False
        for i, name in enumerate(names):
            if any(fnmatchcase(name, p) for p in pats):
                match_count[i] += 1
                all_matches[i].append(group_names[gi])
                if group_idx[i] < 0:
                    group_idx[i] = gi
                    matched_any = True
        if not matched_any:
            warnings.warn(f"group pattern(s) {pats!r} matched no file")

    for i in np.nonzero(match_count > 1)[0]:
        warnings.warn(f"{names[i]}: matches groups {all_matches[i]} - assigned to "
                      f"'{group_names[group_idx[i]]}' (first match wins)")
    unmatched = np.nonzero(group_idx < 0)[0]
    if len(unmatched):
        warnings.warn(f"files matching no group pattern (excluded): "
                      f"{[names[i] for i in unmatched]}")
    return group_names, group_idx


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _selftest():
    ok = True

    # ---------------- sanitize_stem ----------------
    s1 = sanitize_stem('my group!')
    s2 = sanitize_stem('ch488')
    s3 = sanitize_stem('..weird../')
    try:
        sanitize_stem('!!!')
        raises = False
    except ValueError:
        raises = True
    print(f"[stem] 'my group!' -> {s1!r}, 'ch488' -> {s2!r}, '..weird../' -> {s3!r}, "
          f"all-unusable raises: {raises}")
    ok &= s1 == 'my_group' and s2 == 'ch488' and s3 == 'weird' and raises

    # ---------------- assign_groups ----------------
    files = ['a_ch0_01.tif', 'a_ch1_01.tif', 'b_ch0_02.tif', 'b_ch1_02.tif',
             'c_ch2.tif', 'notes.txt']
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        gnames, gidx = assign_groups(files, {'ch0': '*_ch0_*', 'ch1': '*_ch1_*'})
    unmatched_warned = any('no group pattern' in str(w.message) for w in wlist)
    exp_idx = np.array([0, 1, 0, 1, -1, -1])
    print(f"[groups] names {gnames}, idx {gidx.tolist()} (expected {exp_idx.tolist()}), "
          f"unmatched warned: {unmatched_warned}")
    ok &= gnames == ['ch0', 'ch1'] and (gidx == exp_idx).all() and unmatched_warned

    g_all, gi_all = assign_groups(files)  # None -> single default group
    print(f"[groups] default patterns=None: names {g_all}, all idx 0: {bool((gi_all == 0).all())}")
    ok &= g_all == ['ALL'] and (gi_all == 0).all()

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        g2, gi2 = assign_groups(['x_ch0_ch1.tif', 'y.tif'],
                                {'first': '*_ch0*', 'second': ['*_ch1*', '*_ch9*']})
    multi_warned = any('first match wins' in str(w.message) for w in wlist)
    empty_warned = any('matched no file' in str(w.message) for w in wlist)
    print(f"[groups] multi-match: idx {gi2.tolist()} (first wins: {gi2[0] == 0}), "
          f"multi warned: {multi_warned}, empty-pattern warned: {empty_warned}")
    ok &= gi2[0] == 0 and gi2[1] == -1 and multi_warned and empty_warned

    g3, _ = assign_groups(['a.tif'], {'my group!': '*.tif'})
    try:
        assign_groups(['a.tif'], {'my group': '*.tif', 'my_group': '*.tif'})
        collide_raises = False
    except ValueError:
        collide_raises = True
    print(f"[groups] sanitized name {g3}, name collision raises: {collide_raises}")
    ok &= g3 == ['my_group'] and collide_raises

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()
