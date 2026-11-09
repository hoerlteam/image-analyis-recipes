import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L

from calmutils.nn.blocks import ConvBlock, ConvBlockConfig
from calmutils.nn.unet import remap_legacy_state_dict


class SimpleCNNClassifier(nn.Module):
    """
    Simple flexible classification CNN, consisting of conv. blocks (shared calmutils.nn
    ConvBlock) followed by averaging pixels and a linear classification head.

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
                ConvBlock(
                    c_in,
                    c_out,
                    ConvBlockConfig(n_convs=3, residual=True, norm="batch", groups=gi),
                    dimensionality=2,
                )
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

    def load_state_dict(self, state_dict, strict=True):
        # accept checkpoints saved with the legacy ConvBlock layout (flat 'steps' Sequential)
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            if not strict:
                raise
            return super().load_state_dict(remap_legacy_state_dict(state_dict), strict=True)

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
