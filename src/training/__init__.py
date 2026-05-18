"""
training
========
Training loop, loss functions, and visualisation utilities for the CMIP6 SST
pix2pix GAN.

Public API
----------
- Trainer               : Orchestrates GAN training, validation, and checkpointing.
- CombinedLoss          : Weighted L1 + climatology consistency loss for the Generator.
- plot_training_history : Save a 2×2 loss-curve figure to disk.
- plot_predictions      : Visualise Generator predictions vs ground-truth targets.

Example
-------
>>> from src.training import Trainer
>>> import yaml
>>>
>>> with open("config.yaml") as f:
...     cfg = yaml.safe_load(f)
>>>
>>> trainer = Trainer(
...     generator=G,
...     discriminator=D,
...     train_loader=train_loader,
...     val_loader=val_loader,
...     climatology=climatology,
...     cfg=cfg,
...     norm_params=norm_params,
... )
>>> history = trainer.fit()
"""

from .losses import CombinedLoss
from .trainer import Trainer, plot_predictions, plot_training_history

__all__ = [
    "Trainer",
    "CombinedLoss",
    "plot_training_history",
    "plot_predictions",
]
