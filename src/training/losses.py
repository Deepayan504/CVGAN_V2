"""
losses.py
=========
Loss functions for the CMIP6 SST pix2pix GAN.

The generator is trained with a weighted combination of three terms:

1. **Adversarial loss** – standard GAN loss (``BCEWithLogitsLoss``), computed
   in the training loop using the discriminator's patch logits.
2. **L1 reconstruction loss** – pixel-wise mean absolute error between the
   predicted and target SST fields, weighted by ``lambda_l1``.
3. **Climatology consistency loss** – penalises deviation from the long-term
   monthly mean (climatology) of the training data, weighted by
   ``lambda_clim``.  This acts as a soft physical prior that prevents the
   model from predicting climatologically implausible values.

Public API
----------
- CombinedLoss : nn.Module combining L1 and climatology losses.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Weighted L1 + climatology consistency loss for the Generator.

    The adversarial component is handled separately in the training loop
    (via ``BCEWithLogitsLoss`` on the Discriminator outputs).  This module
    computes only the *reconstruction* side of the generator loss.

    Parameters
    ----------
    climatology_data:
        Monthly climatology array of shape ``(12, lat, lon)`` computed from
        the training split.  Each slice ``climatology_data[m]`` is the mean
        SST field for month ``m`` (0 = January … 11 = December).
        Pass a NumPy array; it will be registered as a non-trainable buffer.
    lambda_l1:
        Weight for the L1 reconstruction term.  The pix2pix paper recommends
        ``100.0`` as a strong regulariser that encourages sharp outputs.
    lambda_clim:
        Weight for the climatology consistency term.  Smaller values (e.g.
        ``0.1``) provide a soft physical prior without dominating the L1 loss.

    Notes
    -----
    Both ``lambda_l1`` and ``lambda_clim`` are plain floats, not learnable
    parameters — tune them via ``config.yaml``.
    """

    def __init__(
        self,
        climatology_data: np.ndarray,
        lambda_l1: float = 100.0,
        lambda_clim: float = 0.1,
    ) -> None:
        super().__init__()

        self.lambda_l1   = lambda_l1
        self.lambda_clim = lambda_clim

        # Register climatology as a buffer so it moves with .to(device) and
        # is included in state_dict (for checkpoint portability).
        clim_tensor = torch.from_numpy(climatology_data).float()   # (12, lat, lon)
        self.register_buffer("climatology", clim_tensor)

        logger.info(
            "CombinedLoss — lambda_l1: %.1f  lambda_clim: %.4f  "
            "climatology shape: %s",
            lambda_l1, lambda_clim, tuple(clim_tensor.shape),
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        target_months: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined reconstruction loss.

        Parameters
        ----------
        prediction:
            Generator output of shape ``(B, 1, H, W)``.
        target:
            Ground-truth SST field of shape ``(B, 1, H, W)``.
        target_months:
            Integer month indices for each sample in the batch, shape
            ``(B,)``, values in ``[0, 11]``.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar loss tensor (differentiable).
        loss_dict : dict[str, float]
            Breakdown of individual loss components for logging:
            ``{"l1": ..., "climatology": ..., "total": ...}``.
        """
        # ------------------------------------------------------------------
        # L1 reconstruction loss
        # ------------------------------------------------------------------
        l1_loss = F.l1_loss(prediction, target)

        # ------------------------------------------------------------------
        # Climatology consistency loss
        # ------------------------------------------------------------------
        # Gather the climatology slice for each sample's target month
        # climatology: (12, H, W) → index with target_months → (B, H, W)
        clim_batch = self.climatology[target_months]          # (B, H, W)
        clim_batch = clim_batch.unsqueeze(1)                  # (B, 1, H, W)
        clim_loss  = F.l1_loss(prediction, clim_batch)

        # ------------------------------------------------------------------
        # Weighted sum
        # ------------------------------------------------------------------
        total = self.lambda_l1 * l1_loss + self.lambda_clim * clim_loss

        loss_dict = {
            "l1":          l1_loss.item(),
            "climatology": clim_loss.item(),
            "total":       total.item(),
        }

        return total, loss_dict
