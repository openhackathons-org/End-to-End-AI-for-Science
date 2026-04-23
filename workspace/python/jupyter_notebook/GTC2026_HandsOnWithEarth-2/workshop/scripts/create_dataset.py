import os

import numpy as np
import xarray as xr

# Input paths (raw data)
HRRR_DATA_PATH = "???"
ERA5_DATA_PATH = "???"
INVARIANTS_PATH = "???"

# Output base directory. Cropped zarrs are written under OUT_PATH so they are
# not overwritten. Run compute_stats.py after this to generate stats from these.
# Must match CONUS_DATASET_NAME in training config / data_loader_hrrr_era5.
OUT_PATH = "???"
CONUS_DATASET_NAME = "hrrr"

# Paths written by this script (used by compute_stats.py and data loader)
INVARIANTS_OUT = os.path.join(OUT_PATH, CONUS_DATASET_NAME, "invariants.zarr")
ERA5_OUT = os.path.join(OUT_PATH, "era5.zarr")
HRRR_OUT = os.path.join(OUT_PATH, "hrrr.zarr")

SAN_JOSE_COORDS = (37.3382, -121.8863)


if __name__ == "__main__":
    os.makedirs(OUT_PATH, exist_ok=True)

    # --- Invariants (cropped to Bay Area window) ---
    ds = xr.open_dataset(INVARIANTS_PATH)
    hrrr_lat = ds.latitude.values
    hrrr_lon = ds.longitude.values
    hrrr_index = np.unravel_index(
        np.argmin(
            np.abs(hrrr_lat - SAN_JOSE_COORDS[0])
            + np.abs(hrrr_lon - SAN_JOSE_COORDS[1])
        ),
        hrrr_lat.shape,
    )
    lat_window = slice(hrrr_index[0] - 32, hrrr_index[0] + 32)
    lon_window = slice(hrrr_index[1] - 32, hrrr_index[1] + 32)
    print("Invariants lat_window:", lat_window, "lon_window:", lon_window)
    os.makedirs(os.path.dirname(INVARIANTS_OUT), exist_ok=True)
    invariants_data = ds.isel(y=lat_window, x=lon_window)
    invariants_data.to_zarr(INVARIANTS_OUT, zarr_format=2, mode="w")
    print("Invariant data saved to", INVARIANTS_OUT)

    # --- ERA5 (cropped to Bay Area window) ---
    ds = xr.open_dataset(ERA5_DATA_PATH)
    era5_lat = ds.latitude.values
    era5_lon = ds.longitude.values
    era5_index = np.unravel_index(
        np.argmin(
            np.abs(era5_lat - SAN_JOSE_COORDS[0])
            + np.abs(era5_lon - SAN_JOSE_COORDS[1])
        ),
        era5_lat.shape,
    )
    lat_window = slice(era5_index[0] - 8, era5_index[0] + 8)
    lon_window = slice(era5_index[1] - 8, era5_index[1] + 8)
    era5_data = ds.isel(latitude=lat_window, longitude=lon_window)
    era5_data.to_zarr(ERA5_OUT, zarr_format=2, mode="w")
    print("ERA5 data saved to", ERA5_OUT)

    # --- HRRR (cropped to Bay Area window) ---
    ds = xr.open_dataset(HRRR_DATA_PATH)
    hrrr_lat = ds.latitude.values
    hrrr_lon = ds.longitude.values
    hrrr_index = np.unravel_index(
        np.argmin(
            np.abs(hrrr_lat - SAN_JOSE_COORDS[0])
            + np.abs(hrrr_lon - SAN_JOSE_COORDS[1])
        ),
        hrrr_lat.shape,
    )
    lat_window = slice(hrrr_index[0] - 32, hrrr_index[0] + 32)
    lon_window = slice(hrrr_index[1] - 32, hrrr_index[1] + 32)
    hrrr_data = ds.isel(y=lat_window, x=lon_window)
    hrrr_data.to_zarr(HRRR_OUT, zarr_format=2, mode="w")
    print("HRRR data saved to", HRRR_OUT)