"""Morphological operations on integer label maps."""

import numpy as np
from skimage.measure import regionprops
from skimage.morphology import erosion

from calmutils.morphology.structuring_elements import hypersphere_centered


def shrink_labels(labels, radius):
    """Shrink all objects in a label map by `radius` via per-object erosion.

    Objects are eroded one-by-one, each from its own binary mask, so neighbors
    never interfere: shared interfaces of touching objects erode into both of
    them (like per-object morphological erosion), but labels never bleed into
    each other. Internal holes and the image edge count towards the object
    border (constant padding).

    For expansion (the counterpart of this operation), use
    `skimage.segmentation.expand_labels`, which is EDT-based. Note that the
    two are not exact inverses (structuring element vs. Euclidean distances).
    """
    shrunk = np.zeros_like(labels)
    for region in regionprops(labels):
        eroded = erosion(region.image, hypersphere_centered(labels.ndim, radius), mode="constant")
        shrunk[region.slice][eroded] = region.label
    return shrunk
