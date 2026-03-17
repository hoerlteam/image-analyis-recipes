import numpy as np

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

    if v_high == v_low:
        raise ValueError(f"Quantiles {q_low} and {q_high} are identical "
                         "(value range zero). Cannot normalise.")

    # Linear mapping: (x - v_low) / (v_high - v_low)
    norm_flat = (flat - v_low) / (v_high - v_low)

    if clip:
        norm_flat = np.clip(norm_flat, 0, 1)

    return norm_flat.reshape(arr.shape)