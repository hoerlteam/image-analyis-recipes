# img_utils.py

import numpy as np
from scipy import ndimage

def expand_region(binary_mask, iterations=3):
    """
    Expands a binary region using binary dilation to slightly enlarge cell boundaries.
    """
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    expanded = ndimage.binary_dilation(binary_mask, structure=structure, iterations=iterations)
    return expanded.astype(np.uint8)

def convert_8bit_mask_background(img, mask):
    """
    Normalize a 16-bit or float image to 8-bit based on masked region intensity.
    Background will be normalized separately to preserve contrast.
    """
    img = img.astype(np.float32)
    img_masked = img[mask > 0]

    if img_masked.size != 0:
        vmin, vmax = np.percentile(img_masked, (1, 99))
    else: # no foreground, just normalize min-max
        vmin, vmax = img.min(), img.max()
        
    img_clip = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    img_8bit = (img_clip * 255).astype(np.uint8)
    return img_8bit
