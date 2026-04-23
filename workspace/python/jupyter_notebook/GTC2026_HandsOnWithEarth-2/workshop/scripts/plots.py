from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt


# Bay Area cities for map annotations
BAY_AREA_CITIES = {
    "San Francisco": {"lat": 37.7749, "lon": -122.4194},
    "San Jose": {"lat": 37.3382, "lon": -121.8863},
}


def _to_latlon_2d(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if np.all(np.isnan(lat)) or np.all(np.isnan(lon)):
        raise ValueError("Latitude/longitude are missing or all-NaN.")
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat
    return lat2d, lon2d


def plot_era5_hrrr(
    *,
    era5_path: str | Path,
    hrrr_path: str | Path,
    invariants_path: str | Path,
    idx: int = 0,
    inv_name: str | None = None,
    era5_name: str | None = None,
    hrrr_name: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (18.0, 6.0),
    extent: tuple[float, float, float, float] | None = None,
    show_cities: bool = True,
    show_gridlines: bool = True,
) -> plt.Figure:
    """Plot invariant, ERA5 field, and HRRR field directly from Zarr datasets.
    
    Parameters
    ----------
    era5_path : str or Path
        Path to the ERA5 Zarr dataset.
    hrrr_path : str or Path
        Path to the HRRR Zarr dataset.
    invariants_path : str or Path
        Path to the invariants Zarr dataset.
    idx : int
        Time index to plot.
    inv_name : str, optional
        Name of the invariant field to plot.
    era5_name : str, optional
        Name of the ERA5 channel to plot.
    hrrr_name : str, optional
        Name of the HRRR channel to plot.
    cmap : str
        Colormap to use.
    figsize : tuple
        Figure size (width, height).
    extent : tuple, optional
        Map extent as (lon_min, lon_max, lat_min, lat_max). If None, computed from data.
    show_cities : bool
        Whether to show city markers (San Francisco, San Jose).
    show_gridlines : bool
        Whether to show lat/lon gridlines.
    """
    import xarray as xr

    ds_era5 = xr.open_zarr(era5_path, consolidated=True)
    ds_hrrr = xr.open_zarr(hrrr_path, consolidated=True, mask_and_scale=False)
    ds_invariants = xr.open_zarr(invariants_path)

    era5_names = list(ds_era5.channel.values)
    hrrr_names = list(ds_hrrr.channel.values)
    invariant_names = list(ds_invariants.data_vars)

    era5_name = era5_name or era5_names[0]
    hrrr_name = hrrr_name or hrrr_names[0]
    inv_name = inv_name or invariant_names[0]

    era5_time = ds_era5.time.values[idx]
    hrrr_time = ds_hrrr.time.values[idx]

    era5_field = ds_era5.sel(time=era5_time, channel=era5_name).data.values
    hrrr_field = ds_hrrr.sel(time=hrrr_time, channel=hrrr_name).HRRR.values
    invariant_field = ds_invariants[inv_name].values

    era5_lat2d, era5_lon2d = _to_latlon_2d(
        ds_era5.latitude.values, ds_era5.longitude.values
    )
    hrrr_lat2d, hrrr_lon2d = _to_latlon_2d(
        ds_hrrr.latitude.values, ds_hrrr.longitude.values
    )

    # Compute extent from HRRR grid or use provided extent
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
    else:
        lon_min = float(np.nanmin(hrrr_lon2d))
        lon_max = float(np.nanmax(hrrr_lon2d))
        lat_min = float(np.nanmin(hrrr_lat2d))
        lat_max = float(np.nanmax(hrrr_lat2d))
    
    # Center point for projection
    central_lon = (lon_min + lon_max) / 2
    central_lat = (lat_min + lat_max) / 2

    panels: Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = (
        (f"Invariant: {inv_name}", invariant_field, hrrr_lat2d, hrrr_lon2d),
        (f"ERA5: {era5_name}", era5_field, era5_lat2d, era5_lon2d),
        (f"HRRR: {hrrr_name}", hrrr_field, hrrr_lat2d, hrrr_lon2d),
    )

    # Use Lambert Conformal projection for a nice look
    projection = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
        standard_parallels=(lat_min + 5, lat_max - 5),
    )
    data_crs = ccrs.PlateCarree()
    
    fig, axes = plt.subplots(
        1, 3, figsize=figsize, subplot_kw={"projection": projection},
        constrained_layout=True
    )
    
    for ax, (title, data, lat2d, lon2d) in zip(axes, panels, strict=True):
        # Set extent with padding
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        
        im = ax.pcolormesh(
            lon2d,
            lat2d,
            data,
            transform=data_crs,
            shading="auto",
            cmap=cmap,
        )
        
        # Add map features
        ax.coastlines(resolution="50m", linewidth=1.0, color="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white")
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="white", alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor="lightsteelblue", alpha=0.3)
        ax.add_feature(cfeature.LAKES, facecolor="lightsteelblue", alpha=0.3)
        
        # Add city markers
        if show_cities:
            for city_name, coords in BAY_AREA_CITIES.items():
                ax.plot(
                    coords["lon"], coords["lat"],
                    marker=".", markersize=12, color="red",
                    markeredgecolor="white", markeredgewidth=1.0,
                    transform=data_crs, zorder=10,
                )
                ax.text(
                    coords["lon"] + 0.3, coords["lat"] + 0.2,
                    city_name,
                    transform=data_crs,
                    fontsize=9, fontweight="bold",
                    color="red",
                    zorder=11,
                )
        
        # Add lat/lon gridlines
        if show_gridlines:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", 
                              alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}
        
        ax.set_title(title, fontsize=12, fontweight="bold")
        
        # Add colorbar below each plot with label
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
        # Extract variable name from title (e.g., "ERA5: t2m" -> "t2m")
        var_name = title.split(": ")[-1] if ": " in title else title
        cbar.set_label(var_name, fontsize=10)

    return fig


def create_training_progression_gif(rundir, variable="t2m", sample_idx=0, 
                                    output_path=None, fps=2, loop=0):
    """
    Create an animated GIF showing model improvement across training steps.
    
    Parameters
    ----------
    rundir : str or Path
        Path to the training run directory
    variable : str
        Variable to display (e.g., "t2m", "u10m", "v10m")
    sample_idx : int
        Which validation sample to track across steps
    output_path : str or Path, optional
        Where to save the GIF. If None, saves to rundir/progression_{variable}_{sample_idx}.gif
    fps : int
        Frames per second for the animation
    loop : int
        Number of times to loop (0 = infinite)
    
    Returns
    -------
    Path to the created GIF
    """
    from pathlib import Path
    from PIL import Image
    
    img_dir = Path(rundir) / "images" / variable
    if not img_dir.exists():
        print(f"No images at {img_dir}")
        available = list((Path(rundir) / "images").iterdir()) if (Path(rundir) / "images").exists() else []
        print(f"Available variables: {[v.name for v in available]}")
        return None
    
    # Find all steps and get images for this sample
    all_images = sorted([f for f in img_dir.glob("*.png") if "_spec" not in f.name])
    steps = sorted(set(int(f.stem.split("_")[0]) for f in all_images))
    
    # Get images for this sample across steps
    progression = []
    for step in steps:
        matches = [f for f in all_images if f.stem.startswith(f"{step}_{sample_idx}_")]
        if matches:
            progression.append((step, matches[0]))
    
    if len(progression) < 2:
        print(f"Need at least 2 steps for animation. Found {len(progression)} for sample {sample_idx}")
        samples = sorted(set(int(f.stem.split("_")[1]) for f in all_images))
        print(f"Available sample indices: {samples[:10]}{'...' if len(samples) > 10 else ''}")
        return None
    
    print(f"Creating GIF with {len(progression)} frames (steps: {[s for s, _ in progression]})")
    
    # Load images
    frames = []
    for step, img_path in progression:
        img = Image.open(img_path)
        frames.append(img.copy())
    
    # Set output path
    if output_path is None:
        output_path = Path(rundir) / f"progression_{variable}_{sample_idx}.gif"
    output_path = Path(output_path)
    
    # Save as GIF
    duration = int(1000 / fps)  # ms per frame
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    
    print(f"GIF saved to: {output_path}")
    return output_path

def plot_forecast_evolution(io, variable, figsize=(16, 5)):
    """Plot forecast evolution across all lead times."""
    data = io[variable][0, :, :, :]  # [lead_time, y, x]
    lead_times = io["lead_time"][:]
    n_times = len(lead_times)
    
    fig, axes = plt.subplots(1, n_times, figsize=figsize)
    if n_times == 1:
        axes = [axes]
    
    vmin, vmax = data.min(), data.max()
    
    for i, (ax, lt) in enumerate(zip(axes, lead_times)):
        hours = int(lt / np.timedelta64(1, 'h'))
        im = ax.imshow(data[i], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f't+{hours}h')
        ax.axis('off')
    
    fig.suptitle(f'{variable} Forecast Evolution', fontsize=14, fontweight='bold')
    
    # Add colorbar below plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.25, 0.25, 0.5, 0.05])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    
    return fig