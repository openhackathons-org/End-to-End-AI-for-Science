from datetime import datetime, timedelta
import os

import numpy as np
import xarray as xr
import zarr

from earth2studio.data import ARCO, CDS, NCAR_ERA5


VARIABLES = [
    'u10m', 'v10m', 't2m', 'tcwv', 'sp', 'msl',
    'u1000', 'u850', 'u500', 'u250', 'v1000', 'v850', 'v500', 'v250',
    'z1000', 'z850', 'z500', 'z250', 't1000', 't850', 't500', 't250',
    'q1000', 'q850', 'q500', 'q250'
]


def time_range(start, end, step):
    t = start
    while t < end:
        yield t
        t += step


def create_yearly_zarr(
    zarr_path,
    variables,
    year=2023,
    frequency=timedelta(hours=1),
    zarr_format=2
):
    times = list(time_range(datetime(year, 1, 1), datetime(year+1, 1, 1), frequency))
    times = np.array(times).astype("datetime64[ns]")

    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    lon[lon>=180] -= 360

    group = zarr.open_group(zarr_path, zarr_format=zarr_format)

    def _add_coord(name, data):
        arr = group.create(name, shape=data.shape, dtype=data.dtype, fill_value=None)
        arr[:] = data
        arr.attrs["_ARRAY_DIMENSIONS"] = [name]
        return arr

    _add_coord("time", times)
    _add_coord("channel", np.array(variables))
    _add_coord("latitude", lat)
    _add_coord("longitude", lon)

    name = "data"
    dims = ["time", "channel", "latitude", "longitude"]
    shape = (len(times), len(variables), len(lat), len(lon))
    chunks = (1,) + shape[1:]
    arr = group.create(
        name,
        shape=shape,
        dtype=np.float32,
        chunks=chunks,
        fill_value=None,
        compressors=None
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = dims


def insert_time_step_to_zarr(
    da: xr.DataArray,
    zarr_path: str,
    time_idx_range: int,
    zarr_format: int = 2
):
    g = zarr.open_group(zarr_path, zarr_format=zarr_format)
    x = da.values
    (i0, i1) = time_idx_range
    g["data"][i0:i1,:,:,:] = x


def download_yearly_era5(
    zarr_path,
    year=2023,
    variables=VARIABLES,
    start_time=None,
    end_time=None,
    frequency=timedelta(hours=1),
    download_chunk_size=timedelta(hours=24),
    zarr_format=2,
    data_source="NCAR",
    verbose=False
):
    if not os.path.exists(zarr_path):
        print(f"Creating Zarr archive at {zarr_path}...")
        create_yearly_zarr(
            zarr_path,
            variables=variables,
            year=year,
            frequency=frequency,
            zarr_format=zarr_format
        )

    data_source_cls = {
        "CDS": CDS,
        "ARCO": ARCO,
        "NCAR": NCAR_ERA5
    }[data_source]

    if start_time is None:
        start_time = datetime(year, 1, 1)
    if end_time is None:
        end_time = datetime(year+1, 1, 1)

    def download_chunk(t0, t1):
        t1 = min(t1, end_time)
        start_timestamp = datetime.now()
        print(f"{start_timestamp}: Starting download from {t0} to {t1}...")
        times = time_range(t0, t1, frequency)
        times = np.array(list(times)).astype("datetime64[s]")
        i0 = int((t0 - datetime(year, 1, 1)) / frequency)
        i1 = int((t1 - datetime(year, 1, 1)) / frequency)
        era5_data_source = data_source_cls(cache=False, verbose=verbose, async_timeout=None)
        data = era5_data_source(times, variables)
        insert_time_step_to_zarr(data, zarr_path, (i0, i1), zarr_format=zarr_format)
        end_timestamp = datetime.now()
        print(f"{end_timestamp}: Finished download from {t0} to {t1} in {end_timestamp - start_timestamp}.")

    for t0 in time_range(start_time, end_time, download_chunk_size):
        download_chunk(t0, t0 + download_chunk_size)
