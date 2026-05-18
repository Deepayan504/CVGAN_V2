"""
cmip6_loader.py
===============
Loading and preprocessing utilities for CMIP6 Sea Surface Temperature (SST)
netCDF data.

Typical workflow
----------------
1. ``load_and_process_cmip6_data``  – open the netCDF file, stack time
   dimensions, load the land/sea mask, and split into train / val / test.
2. ``process_with_detrending``      – optionally remove a linear or constant
   trend from each spatial pixel over the time axis.
3. ``normalize_data``               – z-score or min-max scale using statistics
   computed on the training split only.
4. ``denormalize_data``             – convert model output back to °C.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import xarray as xr
from scipy.signal import detrend as scipy_detrend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SSTArray = np.ndarray          # shape (time, lat, lon)
MonthArray = np.ndarray        # shape (time,), values 0–11
NormParams = dict


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_and_process_cmip6_data(
    dataset_path: str,
    mask_path: str = "data/raw/mask.npy",
    n_train: int = 1428,
    n_val: int = 100,
    ensemble_id: int = 0,
) -> tuple[SSTArray, SSTArray, SSTArray,
           MonthArray, MonthArray, MonthArray,
           np.ndarray]:
    """Load and split a CMIP6 SST netCDF file into train / val / test sets.

    The function expects the dataset to contain a variable named ``"sst"``
    with dimensions ``(years, mon, lat, lon)`` — or optionally an additional
    ``ensemble`` dimension.  Years and months are stacked into a single
    ``time`` axis before splitting.

    Parameters
    ----------
    dataset_path:
        Path to the CMIP6 netCDF file (e.g.
        ``"data/raw/EC_Earth3_CC_historical_1850_2014.nc"``).
    mask_path:
        Path to a NumPy ``.npy`` file containing the boolean land/sea mask.
        ``True`` (or ``1``) marks ocean cells.
    n_train:
        Number of monthly time steps used for training.
        Default ``1428`` corresponds to 119 years × 12 months (1850–1968).
    n_val:
        Number of monthly time steps used for validation.
        The remainder becomes the test set.
    ensemble_id:
        Index of the ensemble member to select when the dataset contains
        multiple realisations along an ``"ensemble"`` dimension.

    Returns
    -------
    train_data, val_data, test_data : np.ndarray
        SST arrays of shape ``(n_samples, lat, lon)``.
    train_months, val_months, test_months : np.ndarray
        Integer month indices (0 = January … 11 = December) aligned with
        the corresponding data arrays.
    mask : np.ndarray
        Boolean land/sea mask loaded from *mask_path*.

    Raises
    ------
    KeyError
        If the variable ``"sst"`` is not found in the dataset.
    FileNotFoundError
        If *dataset_path* or *mask_path* do not exist.
    """
    logger.info("Loading dataset from: %s", dataset_path)

    ds = xr.open_dataset(dataset_path)

    if "sst" not in ds:
        raise KeyError(
            f"Variable 'sst' not found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    sst = ds["sst"]

    # Select a single ensemble member if necessary
    if "ensemble" in sst.dims:
        logger.info(
            "Multiple ensembles detected — selecting ensemble index %d.", ensemble_id
        )
        sst = sst.isel(ensemble=ensemble_id)

    # Fill missing (land) values with 0
    sst = sst.fillna(0)

    # Stack (years, mon) → time; result shape: (time, lat, lon)
    sst_time = (
        sst
        .stack(time=("years", "mon"))
        .transpose("time", "lat", "lon")
        .fillna(0)
    )

    data: SSTArray = sst_time.values

    logger.info("Loaded data — shape: %s  range: [%.2f, %.2f]",
                data.shape, data.min(), data.max())

    # ------------------------------------------------------------------
    # Land / sea mask
    # ------------------------------------------------------------------
    mask: np.ndarray = np.load(mask_path)
    logger.info("Mask loaded — shape: %s", mask.shape)

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------
    train_data = data[:n_train]
    val_data   = data[n_train:n_train + n_val]
    test_data  = data[n_train + n_val:]

    logger.info(
        "Split sizes — train: %s  val: %s  test: %s",
        train_data.shape, val_data.shape, test_data.shape,
    )

    # ------------------------------------------------------------------
    # Month index arrays (0 = Jan, …, 11 = Dec)
    # ------------------------------------------------------------------
    total = data.shape[0]
    months_all = _build_month_array(ds, total)

    train_months = months_all[:n_train]
    val_months   = months_all[n_train:n_train + n_val]
    test_months  = months_all[n_train + n_val:]

    return (
        train_data, val_data, test_data,
        train_months, val_months, test_months,
        mask,
    )


def process_with_detrending(
    data: SSTArray,
    detrend_type: Literal["linear", "constant"] | None = "linear",
) -> SSTArray:
    """Remove a trend from each spatial pixel along the time axis.

    Uses a vectorised approach (reshaping to 2-D) rather than a nested
    Python loop, which is significantly faster on large grids.

    Parameters
    ----------
    data:
        SST array of shape ``(time, lat, lon)``.
    detrend_type:
        * ``"linear"``   – remove a least-squares linear fit.
        * ``"constant"`` – subtract the mean (de-mean only).
        * ``None``       – return *data* unchanged.

    Returns
    -------
    np.ndarray
        Detrended array with the same shape as *data*.
    """
    if detrend_type is None:
        return data

    if detrend_type not in ("linear", "constant"):
        raise ValueError(
            f"detrend_type must be 'linear', 'constant', or None; "
            f"got {detrend_type!r}."
        )

    logger.info("Applying %s detrending …", detrend_type)

    n_time, n_lat, n_lon = data.shape

    # Reshape to (time, pixels) so scipy can operate on all pixels at once
    flat = data.reshape(n_time, -1)
    flat_detrended = scipy_detrend(flat, axis=0, type=detrend_type)

    return flat_detrended.reshape(n_time, n_lat, n_lon)


def normalize_data(
    train_data: SSTArray,
    val_data: SSTArray,
    test_data: SSTArray,
    method: Literal["standardize", "minmax", "none"] = "standardize",
) -> tuple[SSTArray, SSTArray, SSTArray, NormParams]:
    """Scale SST data using statistics derived from the training split only.

    Parameters
    ----------
    train_data, val_data, test_data:
        Raw SST arrays of shape ``(n_samples, lat, lon)``.
    method:
        * ``"standardize"`` – zero mean, unit standard deviation (z-score).
        * ``"minmax"``      – scale to the ``[0, 1]`` interval.
        * ``"none"``        – return arrays unchanged.

    Returns
    -------
    train_norm, val_norm, test_norm : np.ndarray
        Scaled arrays.
    norm_params : dict
        Parameters needed to invert the scaling via :func:`denormalize_data`.
    """
    if method == "standardize":
        mean = float(np.mean(train_data))
        std  = float(np.std(train_data))

        if std == 0:
            raise ValueError("Training data has zero standard deviation — cannot standardize.")

        train_norm = (train_data - mean) / std
        val_norm   = (val_data   - mean) / std
        test_norm  = (test_data  - mean) / std

        norm_params: NormParams = {"method": "standardize", "mean": mean, "std": std}
        logger.info("Standardization — mean: %.4f  std: %.4f", mean, std)

    elif method == "minmax":
        min_val = float(np.min(train_data))
        max_val = float(np.max(train_data))
        rng = max_val - min_val

        if rng == 0:
            raise ValueError("Training data has zero range — cannot apply min-max scaling.")

        train_norm = (train_data - min_val) / rng
        val_norm   = (val_data   - min_val) / rng
        test_norm  = (test_data  - min_val) / rng

        norm_params = {"method": "minmax", "min": min_val, "max": max_val}
        logger.info("Min-max scaling — min: %.4f  max: %.4f", min_val, max_val)

    else:  # "none"
        train_norm, val_norm, test_norm = train_data, val_data, test_data
        norm_params = {"method": "none"}
        logger.info("No normalization applied.")

    return train_norm, val_norm, test_norm, norm_params


def denormalize_data(data: SSTArray, norm_params: NormParams) -> SSTArray:
    """Invert normalization to recover physical SST values (°C).

    Parameters
    ----------
    data:
        Normalized SST array (any shape).
    norm_params:
        Dictionary returned by :func:`normalize_data`.

    Returns
    -------
    np.ndarray
        Array in the original physical units.
    """
    method = norm_params.get("method", "none")

    if method == "standardize":
        return data * norm_params["std"] + norm_params["mean"]

    if method == "minmax":
        return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]

    return data  # "none"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_month_array(ds: xr.Dataset, total: int) -> MonthArray:
    """Return a 0-indexed month array (0 = Jan … 11 = Dec) of length *total*.

    Attempts to derive month indices from the dataset's ``"mon"`` coordinate;
    falls back to assuming January start if coordinates are unavailable.

    Parameters
    ----------
    ds:
        Open xarray Dataset (used to inspect coordinate metadata).
    total:
        Total number of time steps needed.

    Returns
    -------
    np.ndarray of shape ``(total,)`` with dtype int, values in ``[0, 11]``.
    """
    if "mon" in ds.coords:
        mon_vals = ds.coords["mon"].values  # e.g. [1, 2, …, 12] or [0, …, 11]
        # Normalise to 0-indexed
        if mon_vals.min() == 1:
            mon_vals = mon_vals - 1
        # Tile to cover all time steps
        n_years = int(np.ceil(total / len(mon_vals)))
        months_all = np.tile(mon_vals, n_years)[:total]
    else:
        logger.warning(
            "'mon' coordinate not found in dataset — assuming data starts in January."
        )
        months_all = np.array([i % 12 for i in range(total)])

    return months_all.astype(int)
