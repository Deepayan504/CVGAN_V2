"""
discriminator.py
================
PatchGAN Discriminator for the CMIP6 SST pix2pix GAN.

Architecture
------------
Rather than producing a single real/fake scalar, the PatchGAN Discriminator
outputs a spatial grid of logits where each value classifies an overlapping
``N × N`` patch of the input as real or fake.  Averaging (or summing) these
logits gives the final discriminator loss.  This design:

* penalises high-frequency artefacts that a global discriminator would miss,
* uses far fewer parameters than a fully-connected head, and
* can be applied to inputs of arbitrary spatial size.

Conditioning
------------
The Discriminator is *conditional*: it receives the concatenation of the
generator input (context months + seasonal encoding) and the target SST field
along the channel axis.  This forces it to evaluate whether the prediction is
consistent with the given input context, not just whether it looks like a
plausible SST map in isolation.

  ``in_channels = num_input_months + 2 + 1``
  (context channels + 1 target/predicted SST channel)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _DiscBlock(nn.Module):
    """Conv2d → (optional BatchNorm) → LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        batch_norm: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1,
                bias=not batch_norm,
            ),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """PatchGAN Discriminator for SST prediction.

    Parameters
    ----------
    in_channels:
        Total input channels fed to the discriminator.  This is the
        *concatenation* of the generator's input and the target (or predicted)
        SST field, i.e. ``num_input_months + 2 + 1``.
    features:
        Base feature width.  Layers grow as ``features → features*2 →
        features*4 → features*8``.  Default ``64``.

    Output
    ------
    A tensor of shape ``(B, 1, H', W')`` containing un-normalised logits.
    Pass through ``BCEWithLogitsLoss`` directly — do **not** apply sigmoid
    before the loss.
    """

    def __init__(
        self,
        in_channels: int = 6,
        features: int = 64,
    ) -> None:
        super().__init__()

        f = features  # shorthand

        self.model = nn.Sequential(
            # Layer 1 — no BN on the first layer (standard pix2pix practice)
            _DiscBlock(in_channels, f,      stride=2, batch_norm=False),
            # Layer 2
            _DiscBlock(f,           f * 2,  stride=2),
            # Layer 3
            _DiscBlock(f * 2,       f * 4,  stride=2),
            # Layer 4 — stride 1 to preserve more spatial resolution
            _DiscBlock(f * 4,       f * 8,  stride=1),
            # Output — single logit per patch
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Concatenated input of shape ``(B, in_channels, H, W)``.
            Construct with ``torch.cat([context, target_or_fake], dim=1)``
            before calling.

        Returns
        -------
        torch.Tensor
            Patch logit map of shape ``(B, 1, H', W')``.
        """
        return self.model(x)
