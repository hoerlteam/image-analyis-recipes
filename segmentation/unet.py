import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import pytorch as L


class ConvBlock(nn.Module):
    """
    Simple module consisting of multiple 3x3 convolutions with LeakyReLUs as activations.

    The first conv maps from input to output channel number and can have stride > 1 to achieve pooling.
    All other convolutions retain the shape of the first intermediate output.

    Optionally, the block can be residual, in which case the intermediate output
    of the first conv is added to the output of the whole block at the end.
    """

    def __init__(self, in_channels, out_channels, dimensionality=2, n_convs=3, stride_1st=1, residual=False, leaky_relu_slope=0.01, batch_norm=False) -> None:
        super().__init__()

        if dimensionality not in (1, 2, 3):
            raise ValueError("Only 1D, 2D, 3D is supported.")

        conv_class = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimensionality]
        batch_norm_class = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[dimensionality]

        padding_mode = "zeros"

        self.residual = residual
        self.batch_norm = batch_norm
        self.steps = []
        for i in range(n_convs):
            if i == 0:
                # first convolution may have stride != 1 to achieve pooling effect
                conv = conv_class(in_channels, out_channels, 3, stride=stride_1st, padding=1, padding_mode=padding_mode)
            else:
                conv = conv_class(out_channels, out_channels, 3, padding=1, padding_mode=padding_mode)
            self.steps.append(conv)

            # add batch norm layer if desired
            if batch_norm:
                self.steps.append(batch_norm_class(out_channels))

            self.steps.append(nn.LeakyReLU(leaky_relu_slope))
        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        if self.residual:
            # apply first conv and activation
            x1 = self.steps[:(3 if self.batch_norm else 2)](x)
            # apply other layers (will be identity if n_convs=1) and add result from first conv
            x = self.steps[(3 if self.batch_norm else 2):](x1) + x1
            return x
        else:
            return self.steps(x)


class UNet(nn.Module):

    def __init__(self, in_channels, intermediate_channels, out_channels, dimensionality=2,
                 residual_conv_blocks=False, n_convs_per_block=3, batch_norm=False):
        super().__init__()

        if dimensionality not in (1, 2, 3):
            raise ValueError("Only 1D, 2D, 3D is supported.")

        conv_class = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimensionality]
        padding_mode = "zeros"

        self.dimensionality = dimensionality

        n_encoder_blocks = len(intermediate_channels)
        self.encoder = {}
        for i, n_channels in enumerate(intermediate_channels):
            if i == 0 or i == n_encoder_blocks - 1:
                # first and last block do not pool
                conv_block = ConvBlock(in_channels if i==0 else intermediate_channels[i-1],
                                         n_channels, dimensionality=dimensionality, batch_norm=batch_norm, residual=residual_conv_blocks, n_convs=n_convs_per_block)
            else:
                # intermediate blocks downsample via stride 2 in first convolution
                conv_block = ConvBlock(intermediate_channels[i-1], n_channels, dimensionality=dimensionality, stride_1st=2,
                                         residual=residual_conv_blocks, n_convs=n_convs_per_block, batch_norm=batch_norm)
            self.encoder[f'encoder_block_{i}'] = conv_block
        self.encoder = nn.ModuleDict(self.encoder)

        self.decoder = {}
        # decoder has one less steps (last intermediate encoder output is "bottom of U")
        for i, n_channels in enumerate(reversed(intermediate_channels[:-1])):
            # input channels are twice the normal size, as we concatenate from skip connections
            conv_block = ConvBlock(n_channels * 2, n_channels, dimensionality=dimensionality, residual=residual_conv_blocks,
                                     n_convs=n_convs_per_block, batch_norm=batch_norm)
            self.decoder[f'decoder_block_{i}'] = conv_block
        self.decoder = nn.ModuleDict(self.decoder)

        # TODO: Batch Norm for up-convolutions & and output conv.

        # up-convolutions are implemented as 1x1 convolution to get correct channel number (followed by interpolation to get correct shape)
        self.up_convs = nn.ModuleList([conv_class(intermediate_channels[n], intermediate_channels[n-1], 1, padding_mode=padding_mode) for n in range(n_encoder_blocks-1, 0, -1)])

        # final 1x1 convolution to get output channel number
        self.out_conv = conv_class(intermediate_channels[0], out_channels, 1, padding_mode=padding_mode)

    def forward(self, x):

        # apply encoder conv blocks, keep intermediate output for skip connections
        encoder_intermediates = []
        for enc in self.encoder.values():
            x = enc(x)
            encoder_intermediates.append(x)

        for i, (upc, dec) in enumerate(zip(self.up_convs, self.decoder.values())):
            # get intermediate output from preceeding encoder layer
            x2 = encoder_intermediates[::-1][i+1]
            # up-convolution: 1x1 followed by interpolation
            x = upc(x)
            x = F.interpolate(x, x2.shape[-self.dimensionality:])
            # concatenate skip connection
            x = torch.cat([x, x2], 1)
            x = dec(x)

        return self.out_conv(x)



class LightningUNet(L.LightningModule):

    """
    Thin wrapper of PyTorch UNet for lightning.
    Only inference is implemented, for training, this should be subclassed.
    """

    def __init__(self, n_classes, intermediate_channels, input_channels=1, unet_kwargs={}, **kwargs):
        
        super().__init__()

        unet_kwargs_default =  {'residual_conv_blocks' :True, 'batch_norm': True}
        unet_kwargs_default.update(unet_kwargs)        
        self.unet = UNet(input_channels, intermediate_channels, n_classes, **unet_kwargs)

        self.save_hyperparameters()

    def forward(self, x):

        # if given a DataLoader batch (collection of tensors) instead of just input, use the first
        if isinstance(x, (tuple, list)):
            x = x[0]
        
        return self.unet(x)
