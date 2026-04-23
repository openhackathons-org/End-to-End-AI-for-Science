"""
Compute per-channel means and stds from the cropped HRRR and ERA5 zarrs produced
by scripts/create_dataset.py. Run this after create_dataset.py.

Writes stats to the layout expected by datasets/data_loader_hrrr_era5.py:
  - <OUT_PATH>/era5/stats/means.npy, stds.npy
  - <OUT_PATH>/<CONUS_DATASET_NAME>/<HRRR_STATS>/means.npy, stds.npy
"""

import os

import numpy as np
import xarray as xr

# Must match the OUT_PATH used in scripts/create_dataset.py (where the zarrs were written)
OUT_PATH = "???"

# Subdir names for stats; must match training config (data_loader_hrrr_era5 expects these)
CONUS_DATASET_NAME = "hrrr"
HRRR_STATS = "stats"

# Paths to the zarrs written by create_dataset.py
ERA5_ZARR_PATH = os.path.join(OUT_PATH, "era5.zarr")
HRRR_ZARR_PATH = os.path.join(OUT_PATH, "hrrr.zarr")


def _get_data_array(ds: xr.Dataset, preferred_var: str | None) -> xr.DataArray:
    """Return the main data array (time + channel + spatial). Prefer preferred_var if present."""
    if preferred_var and preferred_var in ds:
        return ds[preferred_var]
    # Fallback: first data var that has a time dimension
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            return da
    raise ValueError(f"No time-dimension data variable found in dataset: {list(ds.data_vars)}")


def _spatial_dims(da: xr.DataArray) -> list[str]:
    """Return dimension names that are spatial (to reduce over for mean/std)."""
    spatial = {"y", "x", "latitude", "longitude", "lat", "lon"}
    return [d for d in da.dims if d in spatial]


def _open_zarr(path: str) -> xr.Dataset:
    """Open zarr store; try consolidated metadata first (create_dataset may not write it)."""
    try:
        return xr.open_zarr(path, consolidated=True)
    except (KeyError, ValueError):
        return xr.open_zarr(path, consolidated=False)


def main() -> None:
    os.makedirs(OUT_PATH, exist_ok=True)

    # --- ERA5: mean/std over time and space (per channel) ---
    era5_ds = _open_zarr(ERA5_ZARR_PATH)
    era5_da = _get_data_array(era5_ds, "data")
    spatial = _spatial_dims(era5_da)
    reduce_dims = ["time"] + spatial
    era5_mean = era5_da.mean(dim=reduce_dims).values
    era5_std = era5_da.std(dim=reduce_dims).values

    era5_stats_dir = os.path.join(OUT_PATH, "era5", "stats")
    os.makedirs(era5_stats_dir, exist_ok=True)
    np.save(os.path.join(era5_stats_dir, "means.npy"), era5_mean)
    np.save(os.path.join(era5_stats_dir, "stds.npy"), era5_std)
    print("ERA5 stats saved to", era5_stats_dir)

    # --- HRRR: mean/std over time and space (per channel) ---
    hrrr_ds = _open_zarr(HRRR_ZARR_PATH)
    hrrr_da = _get_data_array(hrrr_ds, "HRRR")
    spatial = _spatial_dims(hrrr_da)
    reduce_dims = ["time"] + spatial
    hrrr_mean = hrrr_da.mean(dim=reduce_dims).values
    hrrr_std = hrrr_da.std(dim=reduce_dims).values

    hrrr_stats_dir = os.path.join(OUT_PATH, CONUS_DATASET_NAME, HRRR_STATS)
    os.makedirs(hrrr_stats_dir, exist_ok=True)
    np.save(os.path.join(hrrr_stats_dir, "means.npy"), hrrr_mean)
    np.save(os.path.join(hrrr_stats_dir, "stds.npy"), hrrr_std)
    print("HRRR stats saved to", hrrr_stats_dir)


if __name__ == "__main__":
    main()
