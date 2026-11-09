"""
Backward-compatible re-export shim.

The U-Net implementation now lives in CalmUtils (calmutils.nn): blocks.py (ConvBlock),
unet.py (UNet/UNetConfig/checkpoint helpers), utils.py (compile_if_possible),
lightning_wrappers.py (LightningUNet). Install with: pip install CalmUtils[ml]
"""

from calmutils.nn.blocks import ConvBlock, ConvBlockConfig
from calmutils.nn.unet import (UNet, UNetConfig, save_checkpoint, load_unet_checkpoint,
                              load_state_dict_flexible, remap_legacy_state_dict)
from calmutils.nn.utils import compile_if_possible
from calmutils.nn.lightning_wrappers import LightningUNet, _HAS_LIGHTNING

__all__ = [
    "ConvBlockConfig",
    "UNetConfig",
    "ConvBlock",
    "UNet",
    "LightningUNet",
    "save_checkpoint",
    "load_unet_checkpoint",
    "load_state_dict_flexible",
    "remap_legacy_state_dict",
    "compile_if_possible",
]
