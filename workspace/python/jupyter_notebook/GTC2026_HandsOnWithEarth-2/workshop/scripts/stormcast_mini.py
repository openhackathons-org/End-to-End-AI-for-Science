# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""
StormCastMini: A lightweight subclass of StormCast for educational workshops.

This module provides a simplified version of StormCast that supports:
- Diffusion-only mode (no regression model required)
- Custom variable sets and grid regions
- Loading from workshop-exported model packages

The key insight is that StormCast's parent class handles most of the complexity
(coordinate systems, data fetching, Earth2Studio integration). We only override
what's genuinely different for our workshop models.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr

from earth2studio.data import HRRR, DataSource, ForecastSource, fetch_data
from earth2studio.models.batch import batch_func
from earth2studio.models.px.stormcast import StormCast
from earth2studio.utils import handshake_coords
from earth2studio.utils.coords import map_coords
from earth2studio.utils.type import CoordSystem

from omegaconf import OmegaConf
from physicsnemo.models import Module as PhysicsNemoModule
from physicsnemo.utils.diffusion import deterministic_sampler


class StormCastMini(StormCast):
    """StormCast Mini - A lightweight variant for workshop models.

    Extends StormCast to support:
    - **Diffusion-only mode**: No regression model required
    - **Custom variables**: Train on subset of HRRR channels (e.g., u10m, v10m, t2m)
    - **Flexible conditioning**: Works with or without global conditioning data

    The class inherits coordinate handling (`input_coords`, `output_coords`) and
    Earth2Studio workflow integration from the parent StormCast class.

    Parameters
    ----------
    diffusion_model : torch.nn.Module
        EDM diffusion model for generating predictions
    means, stds : torch.Tensor
        Normalization statistics for state variables [1, C, 1, 1]
    invariants : torch.Tensor
        Static fields (orography, land-sea mask) [1, C_inv, H, W]
    hrrr_lat_lim, hrrr_lon_lim : tuple[int, int]
        HRRR grid index limits defining the regional subset
    variables : list[str]
        State variable names (e.g., ["u10m", "v10m", "t2m"])
    regression_model : torch.nn.Module, optional
        If provided, uses regression + diffusion (like full StormCast)
    conditioning_data_source : DataSource | ForecastSource, optional
        Data source for global conditioning (e.g., ARCO, GFS_FX)
    conditioning_variables : list[str], optional
        Variables to fetch from conditioning data source
    conditioning_means, conditioning_stds : torch.Tensor, optional
        Normalization for conditioning variables
    sampler_args : dict, optional
        EDM sampler configuration (num_steps, sigma_min, sigma_max, etc.)

    Examples
    --------
    >>> # Load a workshop model
    >>> model = StormCastMini.load_model("model_package/", conditioning_data_source=ARCO())
    >>> model = model.to("cuda").eval()
    >>>
    >>> # Run with Earth2Studio workflow
    >>> import earth2studio.run as run
    >>> io = run.deterministic(["2020-08-15"], nsteps=4, model=model, data=HRRR(), io=ZarrBackend())
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        hrrr_lat_lim: tuple[int, int],
        hrrr_lon_lim: tuple[int, int],
        variables: list[str],
        regression_model: torch.nn.Module | None = None,
        conditioning_data_source: DataSource | ForecastSource | None = None,
        conditioning_variables: list[str] | None = None,
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        sampler_args: dict | None = None,
    ):
        # Bypass StormCast.__init__ which requires regression_model
        # Go directly to nn.Module.__init__
        torch.nn.Module.__init__(self)

        # Store models
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model

        # Register normalization buffers
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)
        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

        # Compute grid coordinates from HRRR (same as parent class)
        hrrr_lat, hrrr_lon = HRRR.grid()
        self.lat = hrrr_lat[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.lon = hrrr_lon[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.hrrr_x = HRRR.HRRR_X[hrrr_lon_lim[0] : hrrr_lon_lim[1]]
        self.hrrr_y = HRRR.HRRR_Y[hrrr_lat_lim[0] : hrrr_lat_lim[1]]

        # Variables
        self.variables = np.array(variables)
        self.conditioning_variables = (
            np.array(conditioning_variables) if conditioning_variables else None
        )

        # Data source for conditioning
        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None and conditioning_variables is not None:
            warnings.warn(
                "No conditioning_data_source provided. Set it before inference "
                "or pass conditioning tensor directly to __call__."
            )

        # Sampler configuration
        self.sampler_args = sampler_args or {}

    @torch.inference_mode()
    def _forward(
        self, x: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with diffusion-only support.

        Unlike the parent class, conditioning is optional and regression_model
        can be None. This enables simpler workshop models that only use diffusion.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor [B, C, H, W]
        conditioning : torch.Tensor, optional
            Global conditioning [B, C_cond, H, W], already regridded to HRRR

        Returns
        -------
        torch.Tensor
            Predicted next state [B, C, H, W]
        """
        # Normalize input state
        x_norm = (x - self.means) / self.stds

        # Normalize conditioning if provided
        if conditioning is not None:
            if hasattr(self, "conditioning_means"):
                conditioning = conditioning - self.conditioning_means
            if hasattr(self, "conditioning_stds"):
                conditioning = conditioning / self.conditioning_stds

        # Prepare invariants (repeat for batch)
        invariant_tensor = self.invariants.repeat(x.shape[0], 1, 1, 1)

        # Build diffusion condition based on mode
        if self.regression_model is not None:
            # Regression + Diffusion mode (like parent StormCast)
            if conditioning is not None:
                reg_input = torch.cat((x_norm, conditioning, invariant_tensor), dim=1)
            else:
                reg_input = torch.cat((x_norm, invariant_tensor), dim=1)
            reg_out = self.regression_model(reg_input)
            diff_condition = torch.cat((x_norm, reg_out, invariant_tensor), dim=1)
        else:
            # Diffusion-only mode (workshop default)
            if conditioning is not None:
                diff_condition = torch.cat(
                    (x_norm, conditioning, invariant_tensor), dim=1
                )
            else:
                diff_condition = torch.cat((x_norm, invariant_tensor), dim=1)
            reg_out = torch.zeros_like(x_norm)

        # Run EDM diffusion sampler
        latents = torch.randn_like(x_norm)
        edm_out = deterministic_sampler(
            self.diffusion_model,
            latents=latents,
            img_lr=diff_condition,
            **self.sampler_args,
        )

        # Combine regression + diffusion and denormalize
        out = (reg_out + edm_out) * self.stds + self.means
        return out

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run model for one time step.

        Supports three modes:
        1. **Data source mode**: If `conditioning_data_source` is set, fetches
           and regrids global data automatically (Earth2Studio workflow)
        2. **Direct tensor mode**: If `conditioning` tensor is passed, uses it directly
        3. **No conditioning**: Runs diffusion without global context

        Parameters
        ----------
        x : torch.Tensor
            Input state [B, T, L, C, H, W] (Earth2Studio format)
        coords : CoordSystem
            Coordinate system with time, lead_time, variable, hrrr_y, hrrr_x
        conditioning : torch.Tensor, optional
            Pre-processed conditioning tensor (bypasses data source)

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and updated coordinates
        """
        # Fetch conditioning from data source if needed
        if conditioning is None and self.conditioning_data_source is not None:
            if self.conditioning_variables is None:
                raise ValueError(
                    "conditioning_variables must be set to use conditioning_data_source"
                )

            conditioning, conditioning_coords = fetch_data(
                self.conditioning_data_source,
                time=coords["time"],
                variable=self.conditioning_variables,
                lead_time=coords["lead_time"],
                device=x.device,
                interp_to=coords | {"_lat": self.lat, "_lon": self.lon},
                interp_method="linear",
            )

            # Reorder dimensions to expected format
            conditioning_coords_ordered = OrderedDict(
                {
                    k: conditioning_coords[k]
                    for k in ["time", "lead_time", "variable", "lat", "lon"]
                }
            )
            conditioning, conditioning_coords = map_coords(
                conditioning, conditioning_coords, conditioning_coords_ordered
            )

            # Add batch dimension and validate
            conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)
            handshake_coords(conditioning_coords, coords, "lead_time")
            handshake_coords(conditioning_coords, coords, "time")

        output_coords = self.output_coords(coords)

        # Process each sample in the batch
        for i, _ in enumerate(coords["batch"]):
            for j, _ in enumerate(coords["time"]):
                for k, _ in enumerate(coords["lead_time"]):
                    cond = None
                    if conditioning is not None:
                        cond = conditioning[i, j, k : k + 1]
                    x[i, j, k : k + 1] = self._forward(x[i, j, k : k + 1], cond)

        return x, output_coords

    @classmethod
    def load_model(
        cls,
        package_path: str | Path,
        conditioning_data_source: DataSource | ForecastSource | None = None,
        hrrr_lat_lim: tuple[int, int] = (549, 613),
        hrrr_lon_lim: tuple[int, int] = (157, 221),
    ) -> "StormCastMini":
        """Load StormCastMini from an exported model package.

        Parameters
        ----------
        package_path : str or Path
            Path to exported package directory containing:
            - `model.yaml`: Model configuration
            - `EDMPrecond.*.mdlus`: Diffusion model checkpoint
            - `metadata.zarr.zip`: Normalization stats and variable lists
            - `StormCastUNet.*.mdlus` (optional): Regression model
        conditioning_data_source : DataSource | ForecastSource, optional
            Data source for global conditioning (e.g., ARCO(), GFS_FX())
        hrrr_lat_lim : tuple[int, int]
            HRRR grid latitude limits (default: Bay Area region)
        hrrr_lon_lim : tuple[int, int]
            HRRR grid longitude limits (default: Bay Area region)

        Returns
        -------
        StormCastMini
            Loaded model ready for inference
        """
        package_path = Path(package_path)

        # Register OmegaConf resolver for eval expressions
        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            pass  # Already registered

        # Load configuration
        config = OmegaConf.load(package_path / "model.yaml")

        # Load diffusion model (required)
        diffusion_ckpts = list(package_path.glob("EDMPrecond.*.mdlus"))
        if not diffusion_ckpts:
            raise FileNotFoundError(f"No diffusion checkpoint in {package_path}")
        diffusion = PhysicsNemoModule.from_checkpoint(str(diffusion_ckpts[0]))

        # Load regression model (optional)
        regression = None
        regression_ckpts = list(package_path.glob("StormCastUNet.*.mdlus"))
        if regression_ckpts:
            regression = PhysicsNemoModule.from_checkpoint(str(regression_ckpts[0]))

        # Load metadata
        store = zarr.storage.ZipStore(
            str(package_path / "metadata.zarr.zip"), mode="r"
        )
        metadata = xr.open_zarr(store, zarr_format=2)

        variables = list(metadata["variable"].values)
        conditioning_variables = (
            list(metadata["conditioning_variable"].values)
            if "conditioning_variable" in metadata
            else None
        )

        # Normalization statistics [1, C, 1, 1]
        means = torch.from_numpy(metadata["means"].values[None, :, None, None])
        stds = torch.from_numpy(metadata["stds"].values[None, :, None, None])

        conditioning_means = None
        conditioning_stds = None
        if "conditioning_means" in metadata:
            conditioning_means = torch.from_numpy(
                metadata["conditioning_means"].values[None, :, None, None]
            )
        if "conditioning_stds" in metadata:
            conditioning_stds = torch.from_numpy(
                metadata["conditioning_stds"].values[None, :, None, None]
            )

        # Invariants
        inv_names = config.data.get("invariants", [])
        if inv_names and "invariants" in metadata:
            invariants = metadata["invariants"].sel(invariant=inv_names).values
            invariants = torch.from_numpy(invariants[None, ...])
        else:
            h = hrrr_lat_lim[1] - hrrr_lat_lim[0]
            w = hrrr_lon_lim[1] - hrrr_lon_lim[0]
            invariants = torch.zeros(1, 1, h, w)

        # Sampler args
        sampler_args = dict(config.sampler_args) if config.get("sampler_args") else {}

        return cls(
            diffusion_model=diffusion,
            regression_model=regression,
            means=means,
            stds=stds,
            invariants=invariants,
            hrrr_lat_lim=hrrr_lat_lim,
            hrrr_lon_lim=hrrr_lon_lim,
            variables=variables,
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            sampler_args=sampler_args,
        )
