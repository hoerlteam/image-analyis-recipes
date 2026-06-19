import numpy as np
import warnings

def scale_intensities(arr, in_range, out_range=(0, 1), clip=False):
    """
    Non-clipping (optional) intensity rescaling.
    Implemented using basic array calculations & selections, should work with both NumPy and PyTorch arrays.
    """

    # TODO: move to CalmUtils

    in_low, in_high = in_range

    if in_high == in_low:
        warnings.warn(f"Input range low and high are equal, will result in division-by-zero.")

    # scale to 0-1
    arr = (arr - in_low) / (in_high - in_low)

    # scale to output range if it is not (0-1)
    if out_range != (0, 1):
        out_low, out_high = out_range
        arr = arr * (out_high - out_low) + out_low

    if clip:
        arr[arr < out_low] = out_low
        arr[arr > out_high] = out_high

    return arr
    

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