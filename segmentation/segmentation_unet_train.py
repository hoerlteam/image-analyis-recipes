
from lightning import pytorch as L
import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import sigmoid_focal_loss

from unet import LightningUNet

import os
from glob import glob
from functools import partial
from natsort import natsorted

from torchvision.transforms import v2

from data import SparseLabeledImageDataset, NormalizationStrategy, ZeroChannelDropout


def get_masked_loss_input(logits_pred, y_gt):

    """
    Get loss function input for sparse segmentation.
    Will select from predicted logits and ground-truth labels 
    """

    # mask == 0 indicates unlabelled
    selection = y_gt > 0

    # select logits -> (npix, c) array 
    logits_selected = torch.transpose(logits_pred, 0, 1)[:, selection].T

    # select nonzero from gt mask, subtract 1 (label==1 in sparse indicates background==0, 2 indicates 1, ...)
    y_selected_corrected = y_gt[selection] - 1
    
    return logits_selected, y_selected_corrected



class SparseSegmentationUNet(LightningUNet):

    """
    Training subclass of ligthning UNet for sparse labels
    """

    def training_step(self, batch, batch_idx):
        
        # apply net
        x, y = batch
        yp = self.forward(x)

        # select pixels labeled in GT, calculate CE for those
        logits_selected, y_selected = get_masked_loss_input(yp, y)
        loss = cross_entropy(logits_selected, y_selected)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DenseSegmentationUNet(LightningUNet):
    """
    Training subclass for conventional, full, labels
    """

    def training_step(self, batch, batch_idx):
        return self._train_val_step(batch, 'loss_train')

    def validation_step(self, batch, batch_idx):
        return self._train_val_step(batch, 'loss_val')

    def _train_val_step(self, batch, log_prefix):
        
        # apply net
        x, y = batch
        yp = self.forward(x)

        # only one output channel, use BCE
        if yp.shape[1] == 1:
            yp = yp[:, 0]
            loss = binary_cross_entropy_with_logits(yp, y.float(), reduction='mean')
        # multiple output channels
        else:
            loss = cross_entropy(yp, y)

        
        self.log(log_prefix, loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':

    from math import ceil
    from torch.utils.data import DataLoader

    dataset_paths = [
        {
            'base_path': '/data/agl_data/AndreasMaiser/NSD/26AM06-02_1',
            'image_subfolder': 'patches_gfp+',
            'label_subfolder': 'patches-segmentation-threshold',
            'image_file_pattern': "*_ch0*.tif",
            'label_file_pattern': "*.tif"
        },
        {
            'base_path': '/data/agl_data/AndreasMaiser/NSD/26AM06-02_2',
            'image_subfolder': 'patches_gfp+',
            'label_subfolder': 'patches-segmentation-threshold',
            'image_file_pattern': "*_ch1*.tif",
            'label_file_pattern': "*.tif"
        }
    ]

    # assemble image and label files lists from one or more dataset_paths
    img_files = []
    label_files = []
    for dataset_path in dataset_paths:
        img_files_i = natsorted(glob(os.path.join(dataset_path['base_path'], dataset_path['image_subfolder'], dataset_path['image_file_pattern'])))
        label_files_i = natsorted(glob(os.path.join(dataset_path['base_path'], dataset_path['label_subfolder'], dataset_path['label_file_pattern'])))
        img_files.extend(img_files_i)
        label_files.extend(label_files_i)

    # random resize crop (should work even for smaller img) and flips
    tr = v2.Compose(
        [
            v2.RandomResizedCrop((128,128), scale=(0.9, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            ZeroChannelDropout(keep_idx=2)
        ]
    )

    # plane selector funtions
    # NOTE: we first select labelled planes, than the middle of those
    selectors = [
        # partial(get_labeled_planes_selection, min_labeled_pixels=20),
        # partial(get_mid_planes_selection, q=0.5),
    ]

    dataset_train = SparseLabeledImageDataset(img_files, label_files, transforms=tr,
                                        plane_selectors=selectors,
                                        normalization_strategy=NormalizationStrategy.PER_IMAGE,
                                        plane_sliding_window=5)
    dataset_val = None

    net = DenseSegmentationUNet(1, [64, 128, 128], input_channels=5)
    loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    trainer = L.Trainer(
        logger=L.loggers.CSVLogger(""),
        log_every_n_steps=ceil(len(dataset_train) / loader.batch_size),
        max_epochs=300,
    )

    trainer.fit(net, loader)