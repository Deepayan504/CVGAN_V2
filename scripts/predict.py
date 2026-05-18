"""
scripts/predict.py
==================
Run multi-step rollout inference with a trained checkpoint.

Given an initial sequence of SST months, the generator predicts the next
month, then feeds its own output back as input to predict the month after
that — repeating for ``--steps`` iterations.  This tests the model's ability
to auto-regressively forecast beyond the training window.

Produces:
  - ``outputs/plots/rollout_<N>steps.png`` — spatial SST maps for each
    predicted step, side-by-side with the ground truth.

Usage
-----
    python scripts/predict.py \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --steps      12 \\
        --start-idx  0

The ``--start-idx`` flag selects which test-set sample to use as the
initial seed sequence.
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_and_process_cmip6_data, normalize_data, process_with_detrending
from src.models import Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_prediction(
    generator: torch.nn.Module,
    initial_sequence: np.ndarray,
    start_month: int,
    n_steps: int,
    climatology: np.ndarray,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Auto-regressive multi-step SST rollout.

    At each step the generator predicts the next SST field; the prediction
    is then appended to the sliding window of input months and the oldest
    month is dropped — mimicking real-world iterative forecasting.

    Parameters
    ----------
    generator:
        Trained Generator in eval mode.
    initial_sequence:
        Seed input of shape ``(num_input_months, H, W)`` — normalised SST.
    start_month:
        0-indexed calendar month of the *first* prediction target (0 = Jan).
    n_steps:
        Number of future months to predict.
    climatology:
        Monthly climatology array ``(12, H, W)`` — normalised scale.
    device:
        Torch device.

    Returns
    -------
    predictions : list of np.ndarray
        Each element is a ``(H, W)`` predicted SST field (normalised).
    target_months : list of int
        Calendar month index (0–11) for each predicted step.
    """
    generator.eval()

    num_input_months = initial_sequence.shape[0]
    h, w             = initial_sequence.shape[1], initial_sequence.shape[2]

    # Sliding window — shape (num_input_months, H, W)
    window = initial_sequence.copy()

    predictions: list[np.ndarray]  = []
    target_months: list[int]        = []

    with torch.no_grad():
        for step in range(n_steps):
            target_month = (start_month + step) % 12
            target_months.append(target_month)

            # Seasonal encoding — sin/cos of target month
            sin_enc = np.full((1, h, w), np.sin(2 * np.pi * target_month / 12))
            cos_enc = np.full((1, h, w), np.cos(2 * np.pi * target_month / 12))

            # Climatology slice for the target month
            clim_slice = climatology[target_month][np.newaxis]  # (1, H, W)

            # Build input: (num_input_months + 2, H, W)
            inp = np.concatenate([window, sin_enc, cos_enc], axis=0)
            inp_t = torch.from_numpy(inp).float().unsqueeze(0).to(device)  # (1, C, H, W)

            pred = generator(inp_t)                              # (1, 1, H, W)
            pred_np = pred.squeeze().cpu().numpy()               # (H, W)
            predictions.append(pred_np)

            # Slide the window: drop oldest month, append prediction
            window = np.concatenate([window[1:], pred_np[np.newaxis]], axis=0)

    return predictions, target_months


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rollout(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    target_months: list[int],
    save_path: str,
) -> None:
    """Plot predicted vs ground-truth SST fields for each rollout step.

    Parameters
    ----------
    predictions:
        List of ``(H, W)`` predicted SST arrays.
    ground_truth:
        List of ``(H, W)`` ground-truth SST arrays (same length).
    target_months:
        Calendar month index (0–11) for each step.
    save_path:
        Destination path for the saved figure.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    n_steps = len(predictions)
    fig, axes = plt.subplots(3, n_steps, figsize=(3 * n_steps, 9))

    # Shared colour scale across all frames
    all_vals = np.concatenate([p.ravel() for p in predictions]
                              + [g.ravel() for g in ground_truth])
    vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)

    row_labels = ["Predicted", "Ground Truth", "Difference"]

    for col, (pred, gt, month) in enumerate(zip(predictions, ground_truth, target_months)):
        diff = pred - gt

        for row, (data, cmap) in enumerate(
            [(pred, "RdBu_r"), (gt, "RdBu_r"), (diff, "bwr")]
        ):
            kw = dict(cmap=cmap, origin="lower")
            if row < 2:
                kw.update(vmin=vmin, vmax=vmax)

            im = axes[row, col].imshow(data, **kw)
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(
                    f"Step {col + 1}\n({month_names[month]})", fontsize=9
                )

        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11)

    plt.suptitle(f"Rollout Forecast — {n_steps} steps", fontsize=13, y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rollout plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-regressive rollout inference for the SST GAN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to a .pth checkpoint (best_model.pth).")
    parser.add_argument("--config",    default="config.yaml", metavar="PATH",
                        help="YAML configuration file.")
    parser.add_argument("--steps",     default=12, type=int,
                        help="Number of future months to predict.")
    parser.add_argument("--start-idx", default=0,  type=int, dest="start_idx",
                        help="Index into the test set for the seed sequence.")
    parser.add_argument("--no-cuda",   action="store_true",
                        help="Disable CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})
    plots_dir = paths_cfg.get("plots_dir", "outputs/plots")

    # Device
    device = torch.device(
        "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)

    norm_params      = ckpt.get("norm_params", {"method": "none"})
    climatology      = ckpt.get("climatology")          # (12, H, W) np array
    num_input_months = ckpt.get("num_input_months",
                                model_cfg.get("num_input_months", 3))

    in_channels = num_input_months + 2
    generator   = Generator(
        in_channels=in_channels,
        features=model_cfg.get("generator_features", 128),
        out_channels=model_cfg.get("out_channels", 1),
    ).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()
    logger.info("Generator ready.")

    # ------------------------------------------------------------------
    # Data — test split
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

    _, _, test_norm, _ = normalize_data(
        train_data, val_data, test_data,
        method=norm_params.get("method", "standardize"),
    )

    # ------------------------------------------------------------------
    # Seed sequence
    # ------------------------------------------------------------------
    idx = args.start_idx
    if idx + num_input_months + args.steps > len(test_norm):
        logger.error(
            "start_idx %d + num_input_months %d + steps %d exceeds "
            "test set length %d.",
            idx, num_input_months, args.steps, len(test_norm),
        )
        sys.exit(1)

    initial_sequence = test_norm[idx: idx + num_input_months]    # (T, H, W)
    start_month      = int(test_months[idx + num_input_months])  # month of 1st prediction
    ground_truth     = [
        test_norm[idx + num_input_months + s]
        for s in range(args.steps)
    ]

    logger.info(
        "Rollout: %d steps from test index %d (first target month: %d)",
        args.steps, idx, start_month,
    )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    predictions, target_months = rollout_prediction(
        generator=generator,
        initial_sequence=initial_sequence,
        start_month=start_month,
        n_steps=args.steps,
        climatology=climatology,
        device=device,
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_rollout(
        predictions=predictions,
        ground_truth=ground_truth,
        target_months=target_months,
        save_path=f"{plots_dir}/rollout_{args.steps}steps.png",
    )

    logger.info("Rollout complete.")


if __name__ == "__main__":
    main()
