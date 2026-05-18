"""
data
====
Data loading and preprocessing utilities for CMIP6 SST data.

Public API
----------
- load_and_process_cmip6_data : Load a CMIP6 netCDF file and split into train/val/test.
- process_with_detrending      : Apply linear or constant detrending along the time axis.
- normalize_data               : Standardize or min-max scale the three splits.
- denormalize_data             : Invert normalization back to physical SST units.

Example
-------
>>> from src.data import load_and_process_cmip6_data, normalize_data
>>> splits = load_and_process_cmip6_data("data/raw/EC_Earth3_CC_historical_1850_2014.nc")
>>> train_data, val_data, test_data, *_ = splits
"""

from .cmip6_loader import (
    load_and_process_cmip6_data,
    process_with_detrending,
    normalize_data,
    denormalize_data,
)

__all__ = [
    "load_and_process_cmip6_data",
    "process_with_detrending",
    "normalize_data",
    "denormalize_data",
]
