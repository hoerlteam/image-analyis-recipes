import numpy as np

from skimage.morphology import remove_small_holes, remove_small_objects, closing, h_maxima
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.ndimage import gaussian_filter

from edt import edt


from calmutils.exposure import scale_intensities
from calmutils.morphology.structuring_elements import hypersphere_centered




def normalize_by_quantiles(arr, quantiles, clip=False):
    """
    Scale ``arr`` such that the value at ``quantiles[0]`` percentile becomes 0,
    and the value at ``quantiles[1]`` percentile becomes 1.
    
    Parameters
    ----------
    arr : array_like
        Input data to be normalised.
    quantiles : pair of floats (q_low, q_high)
        lower and upper quantiles for normalization
    clip : bool
        whether to clip result to 0-1. if False (default), values below `q_low` become negative,
        values above `q_high` become greater than one.

    Returns
    -------
    out : ndarray
        Normalised array with the same shape as ``arr``.
    """

    q_low, q_high = quantiles

    arr = np.asarray(arr, dtype=float) # make sure we work on float copy
    flat = arr.ravel()

    # Compute the two reference values
    v_low  = np.quantile(flat, q_low)
    v_high = np.quantile(flat, q_high)

    
    norm_flat = scale_intensities(flat, (v_low, v_high), clip=clip)

    return norm_flat.reshape(arr.shape)


def threshold_segmentation(
    img,
    blur_sigma,
    small_hole_size,
    small_hole_size_perplane,
    small_object_size,
    closing_radius,
    threshold_function=threshold_otsu,
):

    # blur slightly
    if blur_sigma > 0.0:
        img = gaussian_filter(img.astype(float), blur_sigma)

    # get and apply threshold
    segmented = img > threshold_function(img)

    # some morphological cleanup
    if closing_radius > 0:
        segmented = closing(segmented, hypersphere_centered(img.ndim, closing_radius))

    # remove small holes per plane first
    for plane in range(segmented.shape[0]):
        segmented[plane] = remove_small_holes(
            segmented[plane], max_size=small_hole_size_perplane
        )

    segmented = remove_small_holes(segmented, max_size=small_hole_size)
    segmented = remove_small_objects(segmented, max_size=small_object_size)

    return segmented


def edt_watershed_instance_segmentation(mask, h_maxima_threshold, pixel_size):
    dt = edt(mask, anisotropy=pixel_size)
    maxima = h_maxima(dt, h_maxima_threshold)
    segmented_instance = watershed(-dt, label(maxima), mask=mask, connectivity=2)
    return segmented_instance