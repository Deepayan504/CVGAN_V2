"""
weights.py
==========
Weight initialisation utilities for the pix2pix GAN.

The original pix2pix paper initialises all Conv and BatchNorm weights from a
zero-centred Normal distribution (σ = 0.02).  This function is designed to be
passed directly to ``model.apply()``.

Usage
-----
>>> from src.models import Generator, init_weights
>>> G = Generator(in_channels=5, features=128, out_channels=1)
>>> G.apply(init_weights)
"""

from __future__ import annotations

import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """Initialise Conv, ConvTranspose, and BatchNorm weights.

    Follows the pix2pix paper convention:

    * ``Conv2d`` / ``ConvTranspose2d`` weights ~ N(0, 0.02); biases set to 0.
    * ``BatchNorm2d`` weights ~ N(1, 0.02); biases set to 0.

    All other module types are left unchanged.

    Parameters
    ----------
    module:
        Any ``nn.Module`` instance.  Pass to ``model.apply()`` to recurse
        over all sub-modules automatically.

    Example
    -------
    >>> G = Generator(in_channels=5, features=128, out_channels=1)
    >>> G.apply(init_weights)   # returns G with initialised weights
    """
    classname = module.__class__.__name__

    if classname in ("Conv2d", "ConvTranspose2d"):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname == "BatchNorm2d":
        # BN weight (gamma) initialised near 1; bias (beta) at 0
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0.0)
