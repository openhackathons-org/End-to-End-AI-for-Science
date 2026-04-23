from datetime import datetime, timedelta
import gc
import os
import sys

import numcodecs
import numpy as np
import xarray as xr
import zarr

from earth2studio.data import HRRR

VARIABLES = [
    'u10m', 'v10m', 't2m',
]


def time_range(start, end, step):
    t = start
    while t < end:
        yield t
        t += step


def create_yearly_zarr(
    zarr_path,
    latitude,
    longitude,
    variables=VARIABLES,
    year=2023,
    frequency=timedelta(hours=1),
    zarr_format=2
):
    times = list(time_range(datetime(year, 1, 1), datetime(year+1, 1, 1), frequency))
    times = np.array(times).astype("datetime64[ns]")

    y = np.arange(1059)
    x = np.arange(1799)

    group = zarr.open_group(zarr_path, zarr_format=zarr_format)

    def _add_coord(name, data, dims=None):
        arr = group.create(name, shape=data.shape, dtype=data.dtype, fill_value=None)
        arr[:] = data
        arr.attrs["_ARRAY_DIMENSIONS"] = [name] if dims is None else dims
        return arr

    _add_coord("time", times)
    _add_coord("channel", np.array(variables))
    _add_coord("y", y)
    _add_coord("x", x)
    _add_coord("latitude", latitude, ["y", "x"])
    _add_coord("longitude", longitude, ["y", "x"])

    name = "HRRR"
    dims = ["time", "channel", "y", "x"]
    shape = (len(times), len(variables), len(y), len(x))
    chunks = (1,) + shape[1:2] + (1059, 257)
    arr = group.create(
        name,
        shape=shape,
        dtype=np.float32,
        chunks=chunks,
        fill_value=None,
        compressor=numcodecs.zstd.Zstd(level=5)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = dims

    valid_data_arr = group.create("valid_data", shape=len(times), dtype=bool, fill_value=None)
    valid_data_arr[:] = False
    valid_data_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]


def insert_time_step_to_zarr(
    da: xr.DataArray,
    zarr_path: str,
    time_idx_range: int,
    zarr_format: int = 2
):
    g = zarr.open_group(zarr_path, zarr_format=zarr_format)
    x = da.values
    (i0, i1) = time_idx_range
    g["HRRR"][i0:i1,:,:,:] = x
    g["valid_data"][i0:i1] = True


def download_chunk(
    zarr_path,
    t0,
    t1,
    year=2023,
    variables=VARIABLES,
    end_time=None,
    frequency=timedelta(hours=1),
    zarr_format=2,
    verbose=False,
):
    t1 = min(t1, end_time)
    start_timestamp = datetime.now()
    print(f"{start_timestamp}: Starting download from {t0} to {t1}...")
    times = time_range(t0, t1, frequency)
    times = np.array(list(times)).astype("datetime64[s]")
    i0 = int((t0 - datetime(year, 1, 1)) / frequency)
    i1 = int((t1 - datetime(year, 1, 1)) / frequency)
    
    hrrr_data_source = HRRR(cache=False, verbose=verbose, async_timeout=None)

    data = hrrr_data_source(times, variables)
    insert_time_step_to_zarr(data, zarr_path, (i0, i1), zarr_format=zarr_format)
    end_timestamp = datetime.now()
    print(f"{end_timestamp}: Finished download from {t0} to {t1} in {end_timestamp - start_timestamp}.")


def get_hrrr_coordinates(verbose=False):
    """Fetch latitude and longitude grids from HRRR by downloading a single timestep."""
    hrrr_data_source = HRRR(cache=False, verbose=verbose)
    # Download a single timestep to get the coordinate grids
    sample_time = np.array([datetime(2023, 1, 1)]).astype("datetime64[s]")
    sample_data = hrrr_data_source(sample_time, ["t2m"])
    latitude = sample_data.coords["lat"].values
    longitude = sample_data.coords["lon"].values
    return latitude, longitude


def download_yearly_hrrr(
    zarr_path,
    year=2023,
    variables=VARIABLES,
    start_time=None,
    end_time=None,
    frequency=timedelta(hours=1),
    download_chunk_size=timedelta(hours=24),
    zarr_format=2,
    verbose=False,
):
    if not os.path.exists(zarr_path):
        print("Fetching HRRR coordinate grids...")
        latitude, longitude = get_hrrr_coordinates(verbose=verbose)
        print(f"Creating Zarr archive at {zarr_path}...")
        create_yearly_zarr(
            zarr_path,
            latitude=latitude,
            longitude=longitude,
            variables=variables,
            year=year,
            frequency=frequency,
            zarr_format=zarr_format
        )

    if start_time is None:
        start_time = datetime(year, 1, 1)
    if end_time is None:
        end_time = datetime(year+1, 1, 1)

    for t0 in time_range(start_time, end_time, download_chunk_size):
        download_chunk(
            zarr_path,
            t0,
            t0 + download_chunk_size,
            year=year,
            variables=variables,
            end_time=end_time,
            frequency=frequency,
            zarr_format=zarr_format,
            verbose=verbose,
        )
        gc.collect()


def create_era5_region(global_file, region_file, lat, lon, margin=2):
    lat_range = (lat.min(), lat.max())
    lon_range = (lon.min(), lon.max())

    i0 = int(np.floor((90 - lat_range[1]) / 0.25)) - margin
    i1 = int(np.ceil((90 - lat_range[0]) / 0.25)) + margin + 1
    j0 = int(np.floor(lon_range[0] / 0.25)) - margin
    j1 = int(np.ceil(lon_range[1] / 0.25)) + margin + 1

    with xr.open_dataset(global_file, engine='zarr') as ds:
        ds_sliced = ds.isel(latitude=slice(i0, i1), longitude=slice(j0, j1))

        # Reset encoding to ensure fresh chunking for the sliced dataset
        ds_sliced["data"].encoding = {"chunks": (1,) + ds_sliced["data"].shape[1:]}
        for coord in ds_sliced.coords:
            ds_sliced[coord].encoding = {}
        ds_sliced.to_zarr(region_file, zarr_format=2, mode='w')


def decompression_test(zarr_path, time_step=0, repeats=100):
    with xr.open_dataset(zarr_path, engine='zarr', cache=False) as ds:
        start_time = datetime.now()
        for _ in range(repeats):
            x = ds.HRRR[time_step].values
        end_time = datetime.now()

    run_time = (end_time - start_time).total_seconds()
    print(f"{repeats} repeats, time / repeat {run_time/repeats:.3f} s")



if __name__ == "__main__":
    zarr_path = sys.argv[1]
    year = int(sys.argv[2])
    day = int(sys.argv[3])

    start_time = datetime(year, 1, 1) + timedelta(days=day-1)
    end_time = start_time + timedelta(days=1)
    download_yearly_hrrr(
        zarr_path,
        year=year,
        start_time=start_time,
        end_time=end_time,
        download_chunk_size=timedelta(hours=1)
    )
