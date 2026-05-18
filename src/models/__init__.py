"""
models
======
PyTorch model definitions for the CMIP6 SST pix2pix GAN.

Architecture overview
---------------------
The system uses a conditional GAN (cGAN) where:

* :class:`Generator`      – a U-Net that maps ``(input_months + seasonal_encoding)``
                            channels to a single predicted SST field.
* :class:`Discriminator`  – a PatchGAN that classifies overlapping image patches
                            as real or fake, conditioned on the input.

Public API
----------
- Generator      : U-Net generator with skip connections.
- Discriminator  : PatchGAN discriminator.
- init_weights   : Kaiming / normal weight initialiser (call after construction).

Example
-------
>>> from src.models import Generator, Discriminator, init_weights
>>> G = Generator(in_channels=5, features=128, out_channels=1)
>>> D = Discriminator(in_channels=6)
>>> G.apply(init_weights)
>>> D.apply(init_weights)
"""

from .generator import Generator
from .discriminator import Discriminator
from .weights import init_weights

__all__ = [
    "Generator",
    "Discriminator",
    "init_weights",
]
