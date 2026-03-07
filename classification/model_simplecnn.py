import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L


class ConvBlock(nn.Module):
    """
    Simple module consisting of multiple 3x3 convolutions with LeakyReLUs as activations.

    The first conv maps from input to output channel number and can have stride > 1 to achieve pooling.
    All other convolutions retain the shape of the first intermediate output.

    Optionally, the block can be residual, in which case the intermediate output
    of the first conv is added to the output of the whole block at the end.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dimensionality=2,
        n_convs=3,
        stride_1st=1,
        residual=False,
        leaky_relu_slope=0.01,
        batch_norm=False,
        groups=1,
    ) -> None:

        super().__init__()

        if dimensionality not in (1, 2, 3):
            raise ValueError("Only 1D, 2D, 3D is supported.")

        conv_class = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dimensionality]
        batch_norm_class = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[
            dimensionality
        ]

        self.residual = residual
        self.batch_norm = batch_norm
        self.steps = []
        for i in range(n_convs):
            if i == 0:
                # first convolution may have stride != 1 to achieve pooling effect
                conv = conv_class(
                    in_channels,
                    out_channels,
                    3,
                    stride=stride_1st,
                    padding=1,
                    groups=groups,
                )
            else:
                conv = conv_class(
                    out_channels, out_channels, 3, padding=1, groups=groups
                )
            self.steps.append(conv)

            # add batch norm layer if desired
            if batch_norm:
                self.steps.append(batch_norm_class(out_channels))

            self.steps.append(nn.LeakyReLU(leaky_relu_slope))
        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        if self.residual:
            # apply first conv and activation
            x1 = self.steps[: (3 if self.batch_norm else 2)](x)
            # apply other layers (will be identity if n_convs=1) and add result from first conv
            x = self.steps[(3 if self.batch_norm else 2) :](x1) + x1
            return x
        else:
            return self.steps(x)


class SimpleCNNClassifier(nn.Module):
    """
    Simple flexible classification CNN, consisting of U-Net-like conv. blocks
    followed by averaging pixels and a linear classification head.

    Currently 2D-only.
    """

    def __init__(self, in_channels, out_channels, intermediate_channels):
        super().__init__()

        # TODO: make number of conv. groups per block settable
        # grouping reduces number of parameters, but seems slower
        self._ngroups = [1] * len(intermediate_channels)

        self.encoder = []
        for block_idx, (c_in, c_out) in enumerate(
            zip([in_channels] + intermediate_channels, intermediate_channels)
        ):
            gi = self._ngroups[block_idx]

            self.encoder.append(
                ConvBlock(c_in, c_out, residual=True, batch_norm=True, groups=gi)
            )
            # add max pool after each conv block except last
            # -> there, it will be done by adaptive avg pool
            if block_idx != len(intermediate_channels) - 1:
                self.encoder.append(nn.MaxPool2d(2))

        self.encoder = nn.Sequential(*self.encoder)

        # classification head: average remaining pixels to single feature vector, linear to out_channels
        self.fc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(intermediate_channels[-1], out_channels),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_head(x)


class SimpleCNNClassifierLightning(L.LightningModule):
    """
    Lightning wrapper for impleCNNClassifier
    """

    def __init__(self, in_channels, out_channels, intermediate_channels):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleCNNClassifier(in_channels, out_channels, intermediate_channels)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch):
        '''
        common steps for both train and val:
        apply model and get cross-entropy loss
        '''
        img, target = batch
        out = self.model(img)
        loss = F.cross_entropy(out, target)
        return loss, out

    def training_step(self, batch):
        loss, out = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        img, target = batch
        loss, out = self._common_step(batch)

        # calculate val accuracy to display & log
        acc = (out.argmax(dim=1) == target).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer