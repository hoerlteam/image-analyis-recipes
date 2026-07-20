from enum import StrEnum, auto

import numpy as np
import torch
from torchvision.tv_tensors import Image, Mask
from tifffile import imread

from segmentation_utils import scale_intensities


class NormalizationStrategy(StrEnum):
    PER_IMAGE = auto(),
    PER_PLANE = auto(),


# TODO: sparse labelling does not really play a role for this class, rename?
class SparseLabeledImageDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        img_files,
        mask_files,
        transforms=None,
        plane_selectors=None,
        normalization_strategy: NormalizationStrategy=None,
        normalization_quantiles=(0.0, 1.0),
        plane_sliding_window=1
    ):
        """
        PyTorch Dataset of sparsely labelled (0 considered unlabelled) TIF stacks.
        Will only include planes with a minimum number of nonzero pixels.
        
        Parameters
        ----------
        img_files: list of [str/Path]
            input (intensity) image file paths
        mask_files: list of [str/Path]
            mask / label file paths, must match img_files
        transforms: torchvision v2 transform
            transforms to be applied to images. should only be for augmentation
            conversion to tensor is done already
        plane_selectors: list/iterable of callables
            callables that return a selection along the first dimension when applied to a 3D mask
        """

        self.images = []
        self.masks = []
        self.transforms = transforms

        for img_file, mask_file in zip(img_files, mask_files):

            img = imread(img_file)
            mask = imread(mask_file)

            # add dummy z axis for 2D data
            if img.ndim == 2:
                img = img[np.newaxis]
                mask = mask[np.newaxis]

            # normalize intensities (or not)
            # NOTE: do it before conversion to torch as torch.quantile seems to struggle with large arrays
            img = SparseLabeledImageDataset._normalize_intensities(img, normalization_strategy, normalization_quantiles)

            if plane_sliding_window > 1:
                img = sliding_window_planewise_padded(img, plane_sliding_window)

            # apply plane selectors
            # those should be callables that return a selection when applied to the mask
            img_selected, mask_selected = img, mask
            if plane_selectors is not None:
                for selector in plane_selectors:
                    selection = selector(mask_selected)
                    img_selected, mask_selected = img_selected[selection], mask_selected[selection]

            # to torch tensors with standard datatypes
            self.images.extend(torch.from_numpy(img_selected).float())
            self.masks.extend(torch.from_numpy(mask_selected).long())

        # convert to torchvision TVTensors (so augmentations can be easily applied to both img and mask)
        self.images = [Image(img) for img in self.images]
        self.masks = [Mask(mask) for mask in self.masks]

    @staticmethod
    def _normalize_intensities(img, normalization_strategy=None, normalization_quantiles= (0.0, 1.0)):
        
        if normalization_strategy is None:
            return img
        
        elif normalization_strategy == NormalizationStrategy.PER_IMAGE:
            vl, vh = np.quantile(img, normalization_quantiles)
            return scale_intensities(img.astype(np.float32), (vl, vh))
        
        elif normalization_strategy == NormalizationStrategy.PER_PLANE:
            planes = []
            for plane in img:
                vl, vh = np.quantile(plane, normalization_quantiles)
                planes.append( scale_intensities(plane.astype(np.float32), (vl, vh)) )
            return np.stack(planes)
        else:
            raise ValueError('Invalid normalization strategy')

    def __getitem__(self, idx):

        img, mask = self.images[idx], self.masks[idx]

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):

        return len(self.images)


def get_mid_planes_selection(mask, q=0.5):
    """
    Returns a list of interger indices with which the central fraction of planes can be selected.
    """
    
    n_planes = mask.shape[0]
    start = int( (0.5 - q/2) * n_planes )
    stop =  int( (0.5 + q/2) * n_planes )

    # clip to not go oob
    start = max(start, 0)
    stop = min(stop, n_planes)
    
    selection = list(range(start, stop))
    return selection


def get_labeled_planes_selection(mask, min_labeled_pixels=1):
    """
    Get a boolean selection for the subset of xy planes in which there are at least min_labeled_pixels with nonzero label.
    The last two dimensions are interpreted as yx, the result will have shape (N_labeled_planes, Y, X).
    """

    # sum binarized mask over last 2 dimensions, get selection of planes with enough labeled pixels
    mask_bin = mask > 0
    selection = (
        mask_bin.sum(axis=tuple(range(mask.ndim - 2, mask.ndim))) >= min_labeled_pixels
    )

    return selection


def sliding_window_planewise_padded(img, window_size=3, return_copy=True):
    """
    
    (will take form of channels in Conv.Layer input)
    """    
    padding = ((window_size//2, (window_size-1)//2), ) + ((0,0),) * (img.ndim - 1)
    res = np.pad(img, padding)
    res = np.lib.stride_tricks.sliding_window_view(res, window_size, 0)
    res = res.transpose((0, img.ndim) + tuple(range(1, img.ndim)))
    
    return res.copy() if return_copy else res


class ZeroChannelDropout(torch.nn.Module):
    def __init__(self, keep_p=0.5, keep_idx=None):
        """
        Augmentation module that randomly zeros channels in the input.
        Optionally, a specific index can be set to always be kept
        (e.g. when we use this for sliding window of z-positions, we always keep the central one).
        """
        super().__init__()
        
        self.keep_idx = keep_idx
        self.p = keep_p

    def forward(self, *x):
        img, *rest = x

        selection = torch.rand(img.shape[0]) > self.p
        if self.keep_idx is not None:
            selection[self.keep_idx] = True

        img[~selection] = 0
        return img, *rest