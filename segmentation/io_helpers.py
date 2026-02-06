import warnings
from skimage.io import imsave


def imsave_nowarnings(file, img, **kwargs):
    # catch low contrast warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        imsave(file, img, **kwargs)