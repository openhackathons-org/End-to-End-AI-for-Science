# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from scipy.spatial import cKDTree
import dask
import xarray as xr

from .dataset import StormCastDataset


class HrrrEra5DatasetMini(StormCastDataset):
    """
    Minimal paired dataset for time-synchronized ERA5 and HRRR samples.
    
    Uses KNN interpolation to map ERA5 to HRRR grid.
    Expects zarr paths to be provided directly via params.
    
    Required params:
        - era5_path: path to ERA5 zarr file
        - hrrr_path: path to HRRR zarr file
        - invariants_path: path to invariants zarr file
        - era5_stats_path: path to ERA5 stats directory (means.npy, stds.npy)
        - hrrr_stats_path: path to HRRR stats directory (means.npy, stds.npy)
        - era5_channels: list of ERA5 channels or "all"
        - hrrr_channels: list of HRRR channels or "all"
        - invariants: list of invariant channel names
        - dt: time step in hours between input and target
    """

    def __init__(self, params, train):
        dask.config.set(scheduler="synchronous")
        self.params = params
        self.train = train
        self.dt = params.dt
        self.normalize = True

        # Open zarr datasets
        self.ds_era5 = xr.open_zarr(params.era5_path, consolidated=True)
        self.ds_hrrr = xr.open_zarr(params.hrrr_path, consolidated=True, mask_and_scale=False)

        # Get channels from dataset
        all_era5_channels = list(self.ds_era5.channel.values)
        all_hrrr_channels = list(self.ds_hrrr.channel.values)

        # Select which channels to use
        self.era5_channels = (
            all_era5_channels if params.era5_channels == "all" 
            else params.era5_channels
        )
        self.hrrr_channels = (
            all_hrrr_channels if params.hrrr_channels == "all" 
            else params.hrrr_channels
        )

        # Get indices for stats arrays (assumes stats are ordered same as dataset channels)
        era5_idx = [all_era5_channels.index(c) for c in self.era5_channels]
        hrrr_idx = [all_hrrr_channels.index(c) for c in self.hrrr_channels]

        # Load normalization stats
        self.means_era5 = np.load(os.path.join(params.era5_stats_path, "means.npy"))[
            era5_idx, None, None
        ]
        self.stds_era5 = np.load(os.path.join(params.era5_stats_path, "stds.npy"))[
            era5_idx, None, None
        ]
        self.means_hrrr = np.load(os.path.join(params.hrrr_stats_path, "means.npy"))[
            hrrr_idx, None, None
        ]
        self.stds_hrrr = np.load(os.path.join(params.hrrr_stats_path, "stds.npy"))[
            hrrr_idx, None, None
        ]

        # Get coordinates for KNN interpolation
        self.hrrr_lat = self.ds_hrrr.latitude.values
        self.hrrr_lon = self.ds_hrrr.longitude.values
        self.era5_lat = self.ds_era5.latitude.values
        self.era5_lon = self.ds_era5.longitude.values

        # Build KNN tree for ERA5 -> HRRR interpolation
        self._build_knn_interpolator()

        # Get time coordinates - dataset length is (n_times - dt) to allow for target
        self.times = self.ds_hrrr.time.values
        self.n_samples = len(self.times) - self.dt

        # Load invariants
        self.invariant_channels = params.invariants
        if self.invariant_channels:
            self.ds_invariants = xr.open_zarr(params.invariants_path)
            self._invariants = np.stack([self.ds_invariants[c].values for c in self.invariant_channels], axis=0)
        else:
            self._invariants = None

    def _build_knn_interpolator(self):
        """Build KNN tree for interpolating ERA5 to HRRR grid.
        
        ERA5 has 1D lat/lon (regular grid), HRRR has 2D lat/lon (curvilinear grid).
        """
        # ERA5: 1D coords -> create meshgrid to get all (lat, lon) pairs
        era5_lon_2d, era5_lat_2d = np.meshgrid(self.era5_lon, self.era5_lat)
        era5_points = np.column_stack([
            era5_lat_2d.ravel(), 
            era5_lon_2d.ravel()
        ])
        self.knn_tree = cKDTree(era5_points)

        # HRRR: 2D coords -> flatten directly
        hrrr_points = np.column_stack([
            self.hrrr_lat.ravel(), 
            self.hrrr_lon.ravel()
        ])
        _, self.knn_indices = self.knn_tree.query(hrrr_points, k=1)
        self.hrrr_shape = self.hrrr_lat.shape

    def _interp_era5_to_hrrr(self, era5_data):
        """Interpolate ERA5 data to HRRR grid using precomputed KNN indices."""
        # era5_data: (channels, lat, lon)
        n_channels = era5_data.shape[0]
        era5_flat = era5_data.reshape(n_channels, -1)
        
        # Index into flattened ERA5 with KNN indices
        hrrr_data = era5_flat[:, self.knn_indices].reshape(
            n_channels, *self.hrrr_shape
        )
        return hrrr_data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Return data as a dict with background (ERA5) and state (HRRR input/target)."""
        # Get ERA5 background at input time
        era5_inp = self.ds_era5.sel(
            time=self.times[idx], 
            channel=self.era5_channels
        ).data.values
        era5_inp = self._interp_era5_to_hrrr(era5_inp)
        era5_inp = self.normalize_background(era5_inp)

        # Get HRRR input and target
        hrrr_inp = self.ds_hrrr.sel(
            time=self.times[idx], 
            channel=self.hrrr_channels
        ).HRRR.values
        hrrr_tar = self.ds_hrrr.sel(
            time=self.times[idx + self.dt], 
            channel=self.hrrr_channels
        ).HRRR.values

        hrrr_inp = self.normalize_state(hrrr_inp)
        hrrr_tar = self.normalize_state(hrrr_tar)

        return {
            "background": torch.as_tensor(era5_inp),
            "state": (torch.as_tensor(hrrr_inp), torch.as_tensor(hrrr_tar)),
        }

    def normalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from physical units to normalized data."""
        if self.normalize:
            x = (x - self.means_era5) / self.stds_era5
        return x

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from normalized data to physical units."""
        if self.normalize:
            x = x * self.stds_era5 + self.means_era5
        return x

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from physical units to normalized data."""
        if self.normalize:
            x = (x - self.means_hrrr) / self.stds_hrrr
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from normalized data to physical units."""
        if self.normalize:
            x = x * self.stds_hrrr + self.means_hrrr
        return x

    # Required interface methods for StormCastDataset
    def background_channels(self):
        """Metadata for the background channels."""
        return self.era5_channels

    def state_channels(self):
        """Metadata for the state channels."""
        return self.hrrr_channels

    def image_shape(self):
        """Get the (height, width) of the data."""
        return self.hrrr_shape

    def get_invariants(self):
        """Return invariants used for training, or None if no invariants are used."""
        return self._invariants
