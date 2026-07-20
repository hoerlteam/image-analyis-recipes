import json
from math import ceil

import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from lightning import pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from data import load_dataset
from unet import LightningUNet


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
        return self._train_val_step(batch, 'loss_train')

    def validation_step(self, batch, batch_idx):
        return self._train_val_step(batch, 'loss_val')

    def _train_val_step(self, batch, log_prefix):
        
        # apply net
        x, y = batch
        yp = self.forward(x)

        # select pixels labeled in GT, calculate CE for those
        logits_selected, y_selected = get_masked_loss_input(yp, y)
        loss = cross_entropy(logits_selected, y_selected)

        self.log(log_prefix, loss, prog_bar=True)

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


def run_unet_training(dataset_train, dataset_val, dataset_options, network_options):

    # early stop if loss does not decrease for some epochs
    early_stop_callback = EarlyStopping(monitor='loss_val', patience=network_options['early_stop_patience'])

    # save latest model and best "val" loss
    last_model_checkpoint = ModelCheckpoint()
    best_model_checkpoint = ModelCheckpoint(monitor='loss_val', filename='best-{epoch:02d}-{loss_val:.2f}')

    # use either dense or sparse segmentation class (difference is in loss function)
    net_class = SparseSegmentationUNet if dataset_options['sparse_labeling'] else DenseSegmentationUNet

    net = net_class(dataset_options['n_classes'], network_options['unet_intermediate_channels'], dataset_options['plane_sliding_window'])

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)

    trainer = L.Trainer(
        logger=L.loggers.CSVLogger(""),
        callbacks=[last_model_checkpoint, best_model_checkpoint, early_stop_callback],    
        log_every_n_steps=ceil(len(dataset_train) / train_loader.batch_size),
    )

    trainer.fit(net, train_loader, val_loader)



if __name__ == '__main__':

    with open('nucleolin_unet_config.json') as fd:
        config = json.load(fd)

    dataset_paths = config['dataset_paths']
    dataset_options = config['dataset_options']
    network_options = config['network_options']

    dataset_train, dataset_val = load_dataset(dataset_paths, dataset_options)

    run_unet_training(dataset_train, dataset_val, dataset_options, network_options)
    