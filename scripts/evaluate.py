"""
scripts/evaluate.py
===================
Load a saved checkpoint and run evaluation on the test split.

Produces:
  - Console metrics: mean L1, RMSE, and climatology-relative skill score
  - ``outputs/plots/test_predictions.png`` — side-by-side prediction grid
  - ``outputs/plots/error_map.png``        — mean absolute error averaged
                                             over the test set per pixel

Usage
-----
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth

Override dataset or mask path if they differ from the checkpoint's config:

    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --dataset    data/raw/MIROC6_historical_1850_2014.nc \\
        --config     config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_and_process_cmip6_data, normalize_data, process_with_detrending
from src.models import Generator
from src.training import plot_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    generator: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute L1 and RMSE over the entire *data_loader*.

    Parameters
    ----------
    generator:
        Trained Generator in eval mode.
    data_loader:
        DataLoader wrapping the evaluation split.
    device:
        Torch device.

    Returns
    -------
    dict with keys ``"l1"``, ``"rmse"``, and the accumulated
    ``"error_map"`` (mean absolute error per pixel as a NumPy array).
    """
    generator.eval()

    total_l1   = 0.0
    total_mse  = 0.0
    n_samples  = 0
    error_map_accum: np.ndarray | None = None

    with torch.no_grad():
        for batch in data_loader:
            inputs  = batch["input"].to(device)
            targets = batch["target"].to(device)

            preds = generator(inputs)           # (B, 1, H, W)

            diff = (preds - targets).abs()      # (B, 1, H, W)

            total_l1  += diff.sum().item()
            total_mse += (diff ** 2).sum().item()

            # Accumulate spatial error map
            pixel_mae = diff.squeeze(1).cpu().numpy().sum(axis=0)  # (H, W)
            if error_map_accum is None:
                error_map_accum = pixel_mae
            else:
                error_map_accum += pixel_mae

            n_samples += inputs.size(0)

    n_pixels = error_map_accum.size if error_map_accum is not None else 1

    return {
        "l1":        total_l1  / (n_samples * n_pixels),
        "rmse":      float(np.sqrt(total_mse / (n_samples * n_pixels))),
        "error_map": error_map_accum / n_samples if error_map_accum is not None else None,
    }


def plot_error_map(error_map: np.ndarray, save_path: str) -> None:
    """Save a spatial map of the mean absolute error.

    Parameters
    ----------
    error_map:
        Array of shape ``(H, W)`` containing mean absolute error per pixel.
    save_path:
        Destination path for the saved figure.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(error_map, cmap="hot_r", origin="lower")
    plt.colorbar(im, ax=ax, label="Mean Absolute Error")
    ax.set_title("Spatial Error Map — Test Set")
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Error map saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CMIP6 SST GAN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to a .pth checkpoint file (best_model.pth).")
    parser.add_argument("--config",  default="config.yaml", metavar="PATH",
                        help="YAML config file (used for data paths).")
    parser.add_argument("--dataset", default=None, metavar="PATH",
                        help="Override the dataset path from config.")
    parser.add_argument("--n-samples", default=4, type=int, dest="n_samples",
                        help="Number of samples to show in the prediction plot.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if a GPU is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset:
        cfg["paths"]["dataset"] = args.dataset

    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})
    plots_dir = paths_cfg.get("plots_dir", "outputs/plots")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device(
        "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)

    norm_params      = ckpt.get("norm_params", {"method": "none"})
    climatology      = ckpt.get("climatology")
    num_input_months = ckpt.get("num_input_months", model_cfg.get("num_input_months", 3))

    # ------------------------------------------------------------------
    # Rebuild model & load weights
    # ------------------------------------------------------------------
    in_channels  = num_input_months + 2
    features     = model_cfg.get("generator_features", 128)
    out_channels = model_cfg.get("out_channels", 1)

    generator = Generator(
        in_channels=in_channels, features=features, out_channels=out_channels
    ).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()
    logger.info("Generator weights loaded.")

    # ------------------------------------------------------------------
    # Data — test split only
    # ------------------------------------------------------------------
    (
        train_data, val_data, test_data,
        train_months, val_months, test_months,
        _mask,
    ) = load_and_process_cmip6_data(
        dataset_path=paths_cfg["dataset"],
        mask_path=paths_cfg["mask"],
        n_train=data_cfg.get("n_train", 1428),
        n_val=data_cfg.get("n_val", 100),
        ensemble_id=data_cfg.get("ensemble_id", 0),
    )

    detrend_type = data_cfg.get("detrend", None)
    if detrend_type:
        test_data = process_with_detrending(test_data, detrend_type)

    _, _, test_data, _ = normalize_data(
        train_data, val_data, test_data,
        method=norm_params.get("method", "standardize"),
    )

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    try:
        from sst_pix2pix_annual_cycle import SSTDataset
    except ImportError as e:
        logger.error("Could not import SSTDataset: %s", e)
        sys.exit(1)

    test_dataset = SSTDataset(
        sst_data=test_data,
        months=test_months,
        num_input_months=num_input_months,
        climatology=climatology,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get("training", {}).get("batch_size", 16),
        shuffle=False,
        num_workers=2,
    )

    logger.info("Test samples: %d", len(test_dataset))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    logger.info("Running evaluation …")
    results = compute_metrics(generator, test_loader, device)

    logger.info("=" * 50)
    logger.info("Test L1  (normalised): %.6f", results["l1"])
    logger.info("Test RMSE (normalised): %.6f", results["rmse"])
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_predictions(
        generator=generator,
        data_loader=test_loader,
        device=device,
        n_samples=args.n_samples,
        save_path=f"{plots_dir}/test_predictions.png",
    )

    if results["error_map"] is not None:
        plot_error_map(
            error_map=results["error_map"],
            save_path=f"{plots_dir}/error_map.png",
        )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
