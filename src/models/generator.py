"""
generator.py
============
U-Net Generator for the CMIP6 SST pix2pix GAN.

Architecture
------------
The Generator follows the pix2pix U-Net design:

* **Encoder** – a stack of Conv2d → BatchNorm → LeakyReLU blocks that
  progressively halve the spatial resolution while doubling the channel width.
* **Bottleneck** – the deepest encoding layer (no BatchNorm, dropout for
  stochasticity).
* **Decoder** – a mirrored stack of ConvTranspose2d → BatchNorm → ReLU blocks
  that restore the original resolution.  Each decoder layer receives the
  concatenated skip connection from the corresponding encoder layer, preserving
  fine spatial detail.
* **Output head** – a final ConvTranspose2d followed by ``Tanh`` that maps to
  a single-channel SST prediction in the range ``[−1, 1]``.

Input channels
--------------
``in_channels = num_input_months + 2``

The ``+2`` accounts for the sinusoidal seasonal encoding (sin and cos of the
target month) produced by ``get_seasonal_encoding`` in the dataset class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """Conv2d → (optional BatchNorm) → LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not batch_norm),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → (optional Dropout) → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: bool = False,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """U-Net Generator for SST prediction.

    Parameters
    ----------
    in_channels:
        Number of input channels.  Should be ``num_input_months + 2``
        (the ``+2`` is the sinusoidal seasonal encoding).
    features:
        Base number of feature maps in the first encoder layer.  Subsequent
        layers double this up to ``features * 8``, which is then held constant
        for the deeper layers.  Default ``128``.
    out_channels:
        Number of output channels.  ``1`` for a single SST field.

    Notes
    -----
    The spatial resolution must be divisible by ``2 ** depth`` (where *depth*
    is the number of encoder/decoder stages, here 8).  Pad your input if
    necessary before passing it to the model.
    """

    def __init__(
        self,
        in_channels: int = 5,
        features: int = 128,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        f = features  # shorthand

        # ------------------------------------------------------------------
        # Encoder  (each block halves H × W)
        # ------------------------------------------------------------------
        # e1: no BN on the very first layer (standard pix2pix practice)
        self.enc1 = _EncoderBlock(in_channels, f,       batch_norm=False)
        self.enc2 = _EncoderBlock(f,           f * 2)
        self.enc3 = _EncoderBlock(f * 2,       f * 4)
        self.enc4 = _EncoderBlock(f * 4,       f * 8)
        self.enc5 = _EncoderBlock(f * 8,       f * 8)
        self.enc6 = _EncoderBlock(f * 8,       f * 8)
        self.enc7 = _EncoderBlock(f * 8,       f * 8)

        # Bottleneck (deepest layer, no BN, dropout for stochasticity)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Decoder  (each block doubles H × W; skip connections double in_ch)
        # ------------------------------------------------------------------
        self.dec1 = _DecoderBlock(f * 8,      f * 8, dropout=True)
        self.dec2 = _DecoderBlock(f * 8 * 2,  f * 8, dropout=True)
        self.dec3 = _DecoderBlock(f * 8 * 2,  f * 8, dropout=True)
        self.dec4 = _DecoderBlock(f * 8 * 2,  f * 8)
        self.dec5 = _DecoderBlock(f * 8 * 2,  f * 4)
        self.dec6 = _DecoderBlock(f * 4 * 2,  f * 2)
        self.dec7 = _DecoderBlock(f * 2 * 2,  f)

        # Output head
        self.output = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted SST field of shape ``(B, out_channels, H, W)``, values
            in ``[−1, 1]``.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder with skip connections (concatenate along channel dim)
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))

        return self.output(torch.cat([d7, e1], dim=1))
