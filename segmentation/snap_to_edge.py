from skimage.morphology import erosion, dilation
from skimage.segmentation import watershed

from calmutils.morphology.structuring_elements import hypersphere_centered

def snap_labels_to_edge(labels, edge_img, radius_morphology):

    # generate n-dimensional hypersphere as selem
    structuring_element = hypersphere_centered(labels.ndim, radius_morphology)

    # copy original labels, add 1 so background becomes 1
    labels_eroded = labels.copy() + 1
    # set border areas (where dilated and eroded labels are not equal) to 0
    labels_eroded[erosion(labels_eroded, structuring_element) != dilation(labels_eroded, structuring_element)] = 0
    # masked Watershed is much faster
    # we erode the border region once more (so a 1-pixel layer of labels included included) and only do Watershed there
    mask = erosion(labels_eroded, hypersphere_centered(labels.ndim, 1)) == 0
    # fill zeros with watershed, subtract 1 again
    labels_snap = watershed(edge_img, labels_eroded, mask=mask) - 1
    # ignored pixels in watershed are assigned the original value
    labels_snap[~mask] = labels[~mask]

    return labels_snap