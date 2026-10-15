"""Rings (bands) around borders of labelled objects.

Distances are Euclidean pixel distances from the object border, computed via
the Euclidean distance transform (EDT):

- inside: for each pixel of an object, the distance to the nearest pixel that
  does not belong to the same object: background, pixels of other objects
  (shared interfaces, see `interfaces_as_border`), internal holes, or the
  image edge
- outside: for each background pixel, the distance to the nearest object
  pixel; the pixel is assigned the label of the closest object

Following the EDT convention, the outermost pixel of an object and the first
pixel outside of it both have distance 1. Consequently, `start` is clamped to
a minimum of 1 (`start=0` and `start=1` are equivalent).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects


VALID_SIDES = ("inside", "outside", "both")


def _padded_slice(object_slice, shape, pad):
    """Bounding box slice of an object, expanded by `pad` and clipped to the array shape."""
    return tuple(
        slice(max(s.start - pad, 0), min(s.stop + pad, d)) for s, d in zip(object_slice, shape)
    )


def inside_border_distance(labels, interfaces_as_border=True):
    """Distance of each object pixel to the border of its object.

    With `interfaces_as_border=True` (default), any pixel not belonging to the
    same object counts as border: background, pixels of other objects (i.e.
    shared interfaces of touching objects erode into both of them, matching
    per-object morphological erosion), internal holes, and the image edge.
    With `interfaces_as_border=False`, only background (label 0) counts as
    border, so interfaces of touching objects are treated as interior.

    Non-object pixels have distance 0.
    """
    if not interfaces_as_border:
        # distance to the nearest background pixel (single global EDT)
        return distance_transform_edt(labels > 0)

    # per-object EDT: distance to the nearest pixel of a different label
    dist = np.zeros_like(labels, dtype=np.float64)
    for label_value, object_slice in enumerate(find_objects(labels), start=1):
        if object_slice is None:
            continue

        # a single voxel of padding around the bounding box is sufficient:
        # every pixel of that shell is a valid source (nothing of the object
        # lies outside its bbox), and any source further outside is strictly
        # farther from every object pixel than the shell pixel in its direction
        sl = _padded_slice(object_slice, labels.shape, 1)
        object_mask = labels[sl] == label_value
        dist[sl][object_mask] = distance_transform_edt(object_mask)[object_mask]

    return dist


def ring_labels(labels, start=0, end=10, side="inside", interfaces_as_border=True):
    """Label map of the pixels at distance [start, end] from object borders.

    Parameters
    ----------
    labels: integer array with label map (0 = background)
    start: distance (pixels) from the border where the ring starts
        (clamped to >= 1, see module docstring)
    end: distance (pixels) from the border where the ring ends
    side: where to take the ring relative to the border
        - 'inside': band inside the objects
        - 'outside': band outside the objects (labeled by nearest object)
        - 'both': bands on both sides of the border
    interfaces_as_border: whether shared interfaces of touching objects count
        towards the object border (see `inside_border_distance`)

    Returns
    -------
    integer array of the same shape as `labels`, containing the labels of the
    respective objects in the ring pixels and 0 everywhere else. For
    side='both', the inside and outside bands are disjoint (they lie strictly
    inside / outside the objects), so both can simply be combined.

    Notes
    -----
    - Distances are Euclidean (EDT), not morphological ball radii; results can
      differ slightly from erosion/dilation-based rings.
    - Internal holes of objects count towards the object border.
    - The image edge counts towards the object border (labels are padded with
      one background voxel).
    - For small objects, the inside band can be empty (objects fully eroded).
    """
    labels = np.asarray(labels)

    if side not in VALID_SIDES:
        raise ValueError(f"Invalid side {side!r}, available options: {VALID_SIDES}")

    # EDT: outermost object pixel / first outside pixel have distance 1
    start = max(int(start), 1)
    end = int(end)
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    # pad with a one-voxel background layer so object borders at the image edge are detected
    labels_padded = np.pad(labels, 1, mode="constant", constant_values=0)
    inner = tuple(slice(1, -1) for _ in range(labels.ndim))

    result = np.zeros_like(labels)

    if side in ("inside", "both"):
        dist_in = inside_border_distance(labels_padded, interfaces_as_border)[inner]
        mask = (dist_in >= start) & (dist_in <= end)
        result[mask] = labels[mask]

    if side in ("outside", "both"):
        # for background pixels: distance to the nearest object pixel + label of that object
        dist_out_padded, indices = distance_transform_edt(
            labels_padded == 0, return_indices=True
        )
        dist_out = dist_out_padded[inner]
        nearest_label = labels_padded[tuple(indices)][inner]

        mask = (labels == 0) & (dist_out >= start) & (dist_out <= end)
        result[mask] = nearest_label[mask]

    return result
