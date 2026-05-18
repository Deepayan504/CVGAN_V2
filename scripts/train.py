"""
scripts/train.py
================
Command-line entry point for training the CMIP6 SST pix2pix GAN.

This script wires together all components from ``src/`` and drives the full
pipeline:

  1. Load and validate ``config.yaml``
  2. Set global random seeds for reproducibility
  3. Load & preprocess CMIP6 data
  4. Build DataLoaders
  5. Initialise Generator + Discriminator
  6. Run training via :class:`~src.training.Trainer`
  7. Evaluate on the test set and save prediction plots

Usage
-----
Basic (uses config.yaml in the project root):

    python scripts/train.py

Override any config value on the command line:

    python scripts/train.py --config config.yaml \\
                            --epochs 50          \\
                            --batch-size 8       \\
                            --dataset data/raw/MIROC6_historical_1850_2014.nc \\
                            --checkpoint-dir outputs/checkpoints_MIROC6 \\
                            --no-cuda

Resume from a checkpoint:

    python scripts/train.py --resume outputs/checkpoints/checkpoint_epoch_0100.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the script is invoked
# directly (python scripts/train.py) rather than as a module.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    load_and_process_cmip6_data,
    normalize_data,
    process_with_detrending,
)
from src.models import Discriminator, Generator, init_weights
from src.training import Trainer, plot_predictions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d.", seed)


def load_config(path: str) -> dict:
    """Load and return the YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from: %s", config_path)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Patch the loaded config with any explicit CLI arguments."""
    if args.dataset:
        cfg["paths"]["dataset"] = args.dataset
        logger.info("Dataset overridden → %s", args.dataset)

    if args.mask:
        cfg["paths"]["mask"] = args.mask

    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
        logger.info("num_epochs overridden → %d", args.epochs)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    if args.lr:
        cfg["training"]["lr"] = args.lr

    if args.checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir

    if args.no_cuda:
        cfg["hardware"]["device"] = "cpu"
        logger.info("CUDA disabled — forcing CPU.")

    return cfg


def build_dataset(cfg: dict):
    """Import and construct SSTDataset; deferred to avoid hard dep at import time."""
    # SSTDataset lives in the file that hasn't been refactored yet;
    # import it from wherever it currently resides.
    try:
        from sst_pix2pix_annual_cycle import SSTDataset, compute_climatology, get_seasonal_encoding  # noqa: F401
    except ImportError as e:
        logger.error(
            "Could not import SSTDataset / compute_climatology from "
            "'sst_pix2pix_annual_cycle'. Make sure that module is on your "
            "PYTHONPATH.\n  %s", e,
        )
        sys.exit(1)
    return SSTDataset, compute_climatology


def resume_checkpoint(path: str, generator: torch.nn.Module,
                       discriminator: torch.nn.Module,
                       g_optimizer: torch.optim.Optimizer,
                       d_optimizer: torch.optim.Optimizer,
                       device: torch.device) -> tuple[int, dict]:
    """Load a training checkpoint and restore model + optimiser state.

    Parameters
    ----------
    path:
        Path to the ``.pth`` checkpoint file.
    generator, discriminator:
        Model instances (already on *device*).
    g_optimizer, d_optimizer:
        Optimiser instances.
    device:
        Target device.

    Returns
    -------
    start_epoch : int
        The epoch *after* the checkpoint (resume from here).
    history : dict
        Training history recorded up to the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    history     = ckpt.get("history", {})
    logger.info("Resumed from checkpoint: %s  (epoch %d)", path, ckpt["epoch"])
    return start_epoch, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CMIP6 SST pix2pix GAN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="Path to the YAML configuration file.",
    )

    # Data overrides
    parser.add_argument("--dataset",    default=None, metavar="PATH",
                        help="Override paths.dataset in config.")
    parser.add_argument("--mask",       default=None, metavar="PATH",
                        help="Override paths.mask in config.")

    # Training overrides
    parser.add_argument("--epochs",     default=None, type=int,
                        help="Override training.num_epochs.")
    parser.add_argument("--batch-size", default=None, type=int, dest="batch_size",
                        help="Override training.batch_size.")
    parser.add_argument("--lr",         default=None, type=float,
                        help="Override training.lr.")

    # Output
    parser.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir",
                        metavar="DIR",
                        help="Override paths.checkpoint_dir.")

    # Resume
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="Path to a checkpoint to resume training from.")

    # Hardware
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if a GPU is available.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    data_cfg     = cfg.get("data", {})
    model_cfg    = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    paths_cfg    = cfg.get("paths", {})
    hw_cfg       = cfg.get("hardware", {})

    # ------------------------------------------------------------------
    # 2. Reproducibility
    # ------------------------------------------------------------------
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------
    # 3. Device
    # ------------------------------------------------------------------
    device_str = hw_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 4. Data loading & preprocessing
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STEP 1 — Loading CMIP6 data")
    logger.info("=" * 65)

    (
        train_data, val_data, test_data,
        train_months, val_months, test_months,
        mask,
    ) = load_and_process_cmip6_data(
        dataset_path=paths_cfg["dataset"],
        mask_path=paths_cfg["mask"],
        n_train=data_cfg.get("n_train", 1428),
        n_val=data_cfg.get("n_val", 100),
        ensemble_id=data_cfg.get("ensemble_id", 0),
    )

    # Optional detrending
    detrend_type = data_cfg.get("detrend", None)
    if detrend_type:
        logger.info("Applying %s detrending …", detrend_type)
        train_data = process_with_detrending(train_data, detrend_type)
        val_data   = process_with_detrending(val_data,   detrend_type)
        test_data  = process_with_detrending(test_data,  detrend_type)

    # Normalisation
    normalize_method = data_cfg.get("normalize", "standardize")
    train_data, val_data, test_data, norm_params = normalize_data(
        train_data, val_data, test_data, method=normalize_method
    )

    # ------------------------------------------------------------------
    # 5. Climatology + Datasets
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STEP 2 — Building datasets")
    logger.info("=" * 65)

    SSTDataset, compute_climatology = build_dataset(cfg)

    num_input_months = model_cfg.get("num_input_months", 3)
    climatology      = compute_climatology(train_data, train_months)
    logger.info("Climatology shape: %s", climatology.shape)

    train_dataset = SSTDataset(
        sst_data=train_data,
        months=train_months,
        num_input_months=num_input_months,
        climatology=climatology,
    )
    val_dataset = SSTDataset(
        sst_data=val_data,
        months=val_months,
        num_input_months=num_input_months,
        climatology=climatology,
    )
    test_dataset = SSTDataset(
        sst_data=test_data,
        months=test_months,
        num_input_months=num_input_months,
        climatology=climatology,
    )

    pin_memory  = hw_cfg.get("pin_memory", True) and (device.type == "cuda")
    num_workers = training_cfg.get("num_workers", min(4, os.cpu_count() or 1))

    train_loader = DataLoader(
        train_dataset, batch_size=training_cfg.get("batch_size", 16),
        shuffle=True,  num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_cfg.get("batch_size", 16),
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_cfg.get("batch_size", 16),
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    logger.info(
        "Samples — train: %d  val: %d  test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    # ------------------------------------------------------------------
    # 6. Models
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STEP 3 — Initialising models")
    logger.info("=" * 65)

    in_channels = num_input_months + 2      # SST channels + sin/cos seasonal enc
    features    = model_cfg.get("generator_features", 128)
    out_channels = model_cfg.get("out_channels", 1)

    generator     = Generator(in_channels=in_channels,
                               features=features,
                               out_channels=out_channels).to(device)
    discriminator = Discriminator(in_channels=in_channels + 1).to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    total_g = sum(p.numel() for p in generator.parameters())
    total_d = sum(p.numel() for p in discriminator.parameters())
    logger.info("Generator     parameters: %s", f"{total_g:,}")
    logger.info("Discriminator parameters: %s", f"{total_d:,}")

    # ------------------------------------------------------------------
    # 7. Optional resume
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.resume:
        g_opt_tmp = torch.optim.Adam(generator.parameters())
        d_opt_tmp = torch.optim.Adam(discriminator.parameters())
        start_epoch, _ = resume_checkpoint(
            args.resume, generator, discriminator,
            g_opt_tmp, d_opt_tmp, device,
        )
        # Trainer will re-create optimisers from config; just need the
        # weights restored above.

    # ------------------------------------------------------------------
    # 8. Training
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STEP 4 — Training")
    logger.info("=" * 65)

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        climatology=climatology,
        cfg=cfg,
        norm_params=norm_params,
    )

    history = trainer.fit()

    # ------------------------------------------------------------------
    # 9. Test-set evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STEP 5 — Test set evaluation")
    logger.info("=" * 65)

    plots_dir = paths_cfg.get("plots_dir", "outputs/plots")
    plot_predictions(
        generator=generator,
        data_loader=test_loader,
        device=device,
        n_samples=4,
        save_path=os.path.join(plots_dir, "test_predictions.png"),
    )

    logger.info("=" * 65)
    logger.info("All done.  Outputs saved to: %s", paths_cfg.get("checkpoint_dir"))
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
