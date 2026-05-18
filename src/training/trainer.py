"""
trainer.py
==========
Full training and evaluation pipeline for the CMIP6 SST pix2pix GAN.

This module contains:

* :class:`Trainer`            – orchestrates the GAN training loop, validation,
                                checkpointing, and plot generation.
* :func:`plot_training_history` – saves a 2×2 loss-curve figure.
* :func:`plot_predictions`      – visualises a batch of predictions vs targets.

Typical usage
-------------
Instantiate :class:`Trainer` with pre-built DataLoaders and models, then call
:meth:`Trainer.fit` to run the full training loop:

>>> from src.training import Trainer
>>> trainer = Trainer(generator, discriminator, train_loader, val_loader,
...                   climatology, cfg)
>>> trainer.fit()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import CombinedLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """GAN trainer for the CMIP6 SST pix2pix model.

    Parameters
    ----------
    generator:
        U-Net generator (``src.models.Generator``).
    discriminator:
        PatchGAN discriminator (``src.models.Discriminator``).
    train_loader:
        DataLoader for the training split.
    val_loader:
        DataLoader for the validation split.
    climatology:
        Monthly climatology array ``(12, lat, lon)`` computed from training
        data — passed through to :class:`~src.training.losses.CombinedLoss`.
    cfg:
        Flat configuration dictionary (typically loaded from ``config.yaml``).
        Expected keys are documented in ``config.yaml``; all training
        hyperparameters are read from here so no magic numbers live in this
        class.
    norm_params:
        Normalisation parameters dict returned by
        ``src.data.normalize_data``.  Stored in checkpoints for inference.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        climatology: np.ndarray,
        cfg: dict[str, Any],
        norm_params: dict | None = None,
    ) -> None:
        training_cfg = cfg.get("training", {})
        paths_cfg    = cfg.get("paths", {})
        hw_cfg       = cfg.get("hardware", {})

        # ------------------------------------------------------------------
        # Device
        # ------------------------------------------------------------------
        device_str = hw_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        logger.info("Using device: %s", self.device)

        # ------------------------------------------------------------------
        # Models
        # ------------------------------------------------------------------
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.climatology  = climatology
        self.norm_params  = norm_params or {"method": "none"}

        # ------------------------------------------------------------------
        # Hyperparameters
        # ------------------------------------------------------------------
        self.num_epochs     = training_cfg.get("num_epochs", 200)
        self.lr             = training_cfg.get("lr", 0.0002)
        self.beta1          = training_cfg.get("adam_beta1", 0.5)
        self.beta2          = training_cfg.get("adam_beta2", 0.999)
        self.lr_step_size   = training_cfg.get("lr_step_size", 30)
        self.lr_gamma       = training_cfg.get("lr_gamma", 0.5)
        self.lambda_l1      = training_cfg.get("lambda_l1", 100.0)
        self.lambda_clim    = training_cfg.get("lambda_clim", 0.1)
        self.real_smooth    = training_cfg.get("real_label_smooth", 0.9)
        self.fake_smooth    = training_cfg.get("fake_label_smooth", 0.1)
        self.save_every     = training_cfg.get("save_every_n_epochs", 10)
        self.keep_best      = training_cfg.get("keep_best", True)
        self.num_input_months = cfg.get("model", {}).get("num_input_months", 3)

        # ------------------------------------------------------------------
        # Paths
        # ------------------------------------------------------------------
        self.checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.plots_dir      = Path(paths_cfg.get("plots_dir", "outputs/plots"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Loss functions
        # ------------------------------------------------------------------
        self.criterion = CombinedLoss(
            climatology_data=climatology,
            lambda_l1=self.lambda_l1,
            lambda_clim=self.lambda_clim,
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

        # ------------------------------------------------------------------
        # Optimisers & schedulers
        # ------------------------------------------------------------------
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.g_scheduler = torch.optim.lr_scheduler.StepLR(
            self.g_optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        self.d_scheduler = torch.optim.lr_scheduler.StepLR(
            self.d_optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )

        # Training history
        self.history: dict[str, list[float]] = {
            "train_g_loss":    [],
            "train_d_loss":    [],
            "train_l1_loss":   [],
            "train_clim_loss": [],
            "val_g_loss":      [],
            "val_l1_loss":     [],
        }

        self._best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns
        -------
        dict[str, list[float]]
            Training history with keys:
            ``train_g_loss``, ``train_d_loss``, ``train_l1_loss``,
            ``train_clim_loss``, ``val_g_loss``, ``val_l1_loss``.
        """
        logger.info("Starting training for %d epochs.", self.num_epochs)

        for epoch in range(self.num_epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch()

            self._update_history(train_metrics, val_metrics)
            self._log_epoch(epoch, train_metrics, val_metrics)

            self.g_scheduler.step()
            self.d_scheduler.step()

            self._maybe_save_checkpoint(epoch, val_metrics["g_loss"])

        plot_training_history(
            self.history,
            save_path=str(self.plots_dir / "training_history.png"),
        )
        logger.info("Training complete.  Best val loss: %.4f", self._best_val_loss)
        return self.history

    # ------------------------------------------------------------------
    # Private — epoch-level
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return averaged metrics."""
        self.G.train()
        self.D.train()

        totals = dict(g_loss=0.0, d_loss=0.0, l1_loss=0.0, clim_loss=0.0)
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            inputs        = batch["input"].to(self.device)
            targets       = batch["target"].to(self.device)
            target_months = batch["target_month"].to(self.device)

            d_loss          = self._train_discriminator(inputs, targets)
            g_loss, l1, cl  = self._train_generator(inputs, targets, target_months)

            totals["g_loss"]    += g_loss
            totals["d_loss"]    += d_loss
            totals["l1_loss"]   += l1
            totals["clim_loss"] += cl

            if batch_idx % 50 == 0:
                logger.info(
                    "Epoch %d/%d  [%d/%d]  G: %.4f  D: %.4f  L1: %.4f  Clim: %.4f",
                    epoch + 1, self.num_epochs, batch_idx, n_batches,
                    g_loss, d_loss, l1, cl,
                )

        return {k: v / n_batches for k, v in totals.items()}

    def _val_epoch(self) -> dict[str, float]:
        """Run one validation epoch and return averaged metrics."""
        self.G.eval()
        totals    = dict(g_loss=0.0, l1_loss=0.0)
        n_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                inputs        = batch["input"].to(self.device)
                targets       = batch["target"].to(self.device)
                target_months = batch["target_month"].to(self.device)

                fake_outputs           = self.G(inputs)
                g_recon, loss_dict = self.criterion(fake_outputs, targets, target_months)

                totals["g_loss"]  += g_recon.item()
                totals["l1_loss"] += loss_dict["l1"]

        return {k: v / n_batches for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Private — batch-level
    # ------------------------------------------------------------------

    def _train_discriminator(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        self.d_optimizer.zero_grad()

        # Real pair
        real_pair  = torch.cat([inputs, targets], dim=1)
        d_real     = self.D(real_pair)
        real_labels = torch.ones_like(d_real) * self.real_smooth
        d_real_loss = self.bce_loss(d_real, real_labels)

        # Fake pair (detach so gradients don't flow into G)
        with torch.no_grad():
            fake_outputs = self.G(inputs)
        fake_pair  = torch.cat([inputs, fake_outputs], dim=1)
        d_fake     = self.D(fake_pair)
        fake_labels = torch.zeros_like(d_fake) + self.fake_smooth
        d_fake_loss = self.bce_loss(d_fake, fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def _train_generator(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        target_months: torch.Tensor,
    ) -> tuple[float, float, float]:
        self.g_optimizer.zero_grad()

        fake_outputs = self.G(inputs)
        fake_pair    = torch.cat([inputs, fake_outputs], dim=1)
        d_fake       = self.D(fake_pair)

        # Generator wants discriminator to output 1 (real) for its fakes
        real_labels  = torch.ones_like(d_fake)
        g_adv_loss   = self.bce_loss(d_fake, real_labels)

        g_recon, loss_dict = self.criterion(fake_outputs, targets, target_months)

        g_loss = g_adv_loss + g_recon
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), loss_dict["l1"], loss_dict["climatology"]

    # ------------------------------------------------------------------
    # Private — bookkeeping
    # ------------------------------------------------------------------

    def _update_history(
        self,
        train: dict[str, float],
        val: dict[str, float],
    ) -> None:
        self.history["train_g_loss"].append(train["g_loss"])
        self.history["train_d_loss"].append(train["d_loss"])
        self.history["train_l1_loss"].append(train["l1_loss"])
        self.history["train_clim_loss"].append(train["clim_loss"])
        self.history["val_g_loss"].append(val["g_loss"])
        self.history["val_l1_loss"].append(val["l1_loss"])

    def _log_epoch(
        self,
        epoch: int,
        train: dict[str, float],
        val: dict[str, float],
    ) -> None:
        logger.info(
            "Epoch %d/%d — Train G: %.4f  D: %.4f | Val G: %.4f  L1: %.4f",
            epoch + 1, self.num_epochs,
            train["g_loss"], train["d_loss"],
            val["g_loss"],   val["l1_loss"],
        )

    def _maybe_save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save best-model and periodic checkpoints."""
        state = {
            "epoch":                    epoch,
            "generator_state_dict":     self.G.state_dict(),
            "discriminator_state_dict": self.D.state_dict(),
            "g_optimizer_state_dict":   self.g_optimizer.state_dict(),
            "d_optimizer_state_dict":   self.d_optimizer.state_dict(),
            "history":                  self.history,
            "climatology":              self.climatology,
            "norm_params":              self.norm_params,
            "num_input_months":         self.num_input_months,
        }

        # Best model
        if self.keep_best and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            path = self.checkpoint_dir / "best_model.pth"
            torch.save({**state, "best_val_loss": self._best_val_loss}, path)
            logger.info("✓ Best model saved → %s  (val_loss: %.4f)", path, val_loss)

        # Periodic checkpoint
        if (epoch + 1) % self.save_every == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth"
            torch.save(state, path)
            logger.info("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    save_path: str = "outputs/plots/training_history.png",
) -> None:
    """Save a 2×2 grid of training and validation loss curves.

    Parameters
    ----------
    history:
        Dictionary returned by :meth:`Trainer.fit`.
    save_path:
        Full path (including filename) where the figure is saved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generator loss — train + val
    axes[0, 0].plot(history["train_g_loss"], label="Train")
    axes[0, 0].plot(history["val_g_loss"],   label="Val")
    axes[0, 0].set_title("Generator Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Discriminator loss — train only
    axes[0, 1].plot(history["train_d_loss"], color="tab:orange")
    axes[0, 1].set_title("Discriminator Loss (Train)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    # L1 loss — train + val
    axes[1, 0].plot(history["train_l1_loss"], label="Train")
    axes[1, 0].plot(history["val_l1_loss"],   label="Val")
    axes[1, 0].set_title("L1 Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Climatology loss — train only
    axes[1, 1].plot(history["train_clim_loss"], color="tab:green")
    axes[1, 1].set_title("Climatology Loss (Train)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training history plot saved → %s", save_path)


def plot_predictions(
    generator: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    n_samples: int = 4,
    save_path: str = "outputs/plots/predictions.png",
) -> None:
    """Visualise generator predictions against ground-truth targets.

    Produces a three-row figure for *n_samples* examples:

    * Row 1 – last input month (context)
    * Row 2 – ground-truth target SST
    * Row 3 – generator prediction

    Parameters
    ----------
    generator:
        Trained Generator in eval mode.
    data_loader:
        DataLoader to sample from (typically the test loader).
    device:
        Torch device string or object.
    n_samples:
        Number of examples to plot side-by-side.
    save_path:
        Destination path for the saved figure.
    """
    generator.eval()
    device = torch.device(device)

    batch = next(iter(data_loader))
    inputs        = batch["input"][:n_samples].to(device)
    targets       = batch["target"][:n_samples].to(device)

    with torch.no_grad():
        predictions = generator(inputs)

    # Move to CPU NumPy for plotting
    inputs_np      = inputs.cpu().numpy()       # (N, C, H, W)
    targets_np     = targets.cpu().numpy()      # (N, 1, H, W)
    predictions_np = predictions.cpu().numpy()  # (N, 1, H, W)

    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 10))

    row_labels = ["Input (last month)", "Target SST", "Predicted SST"]

    for col in range(n_samples):
        # Last input channel as context
        axes[0, col].imshow(inputs_np[col, -1], cmap="RdBu_r", origin="lower")
        axes[0, col].set_title(f"Sample {col + 1}")
        axes[0, col].axis("off")

        # Target
        axes[1, col].imshow(targets_np[col, 0], cmap="RdBu_r", origin="lower")
        axes[1, col].axis("off")

        # Prediction
        axes[2, col].imshow(predictions_np[col, 0], cmap="RdBu_r", origin="lower")
        axes[2, col].axis("off")

    # Row labels on the left-most column
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Predictions plot saved → %s", save_path)
