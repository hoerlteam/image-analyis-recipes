import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity


def get_segmentation_visualization(labels, img, visualization_quantiles=(0.02, 0.9995)):
    
    # max projection if necessary
    if img.ndim == 3:
        img = img.max(axis=0)
        labels = labels.max(axis=0)
    
    # rescale to 8bit range
    img_rescaled = rescale_intensity(
        img,
        in_range=tuple(np.quantile(img, visualization_quantiles)),
        out_range="uint8",
    )
    
    # overlay labels
    # NOTE: bg_label is 0 in our binary segmentation
    rgb_visualization = label2rgb(labels, img_rescaled, bg_label=0)
    # float result to 8bit uint
    rgb_visualization = (rgb_visualization * 255).astype(np.uint8)

    return rgb_visualization