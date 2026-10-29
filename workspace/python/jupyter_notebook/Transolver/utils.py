"""Shared utilities for the seven Transolver and GeoTransolver tutorials.

This is the single companion Python file used by the notebooks. The sections
below cover geometry preparation, teaching figures, CFD surface rendering,
training diagnostics, field/force calculations, and calibrated uncertainty.

PyTorch and PyVista are imported inside the helpers that need them. Results,
training support, and checkpoints are supplied separately; this file contains
no model weights. Plot helpers display and return their figures where relevant.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv
from tabulate import tabulate
from types import SimpleNamespace


# ===========================================================================
# Shared geometry frame and velocity conditioning
# ===========================================================================

"""Auditable geometry scaling and velocity-only conditioning for surface models.

This module needs only NumPy and the Python standard library. The fixed shared
scale comes from the union bounding box of the actual stored STL geometries.
Every case is centered by its own unweighted mean STL triangle center, matching
the existing datapipe convention. Normals and native pressure/shear targets are
not changed by these coordinate helpers.
"""




SCHEMA_VERSION = 1
CENTERING = "per_case_mean_stl_triangle_centers"
SCALING = "global_union_stl_bbox_full_side_lengths"
_STL_TRIANGLE = np.dtype([
    ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
])


def binary_stl_summary(path, *, triangle_chunk_size=65536):
    """Stream one binary STL once, returning exact vertex bounds and SHA-256.

    Only one triangle chunk is held in memory. An explicit file-length check
    rejects ASCII STL, truncation, and trailing data rather than interpreting
    an unsupported format. The reported triangle-center mean is a float64 audit
    statistic; training continues to calculate the mean of its stored centers.
    """
    path = Path(path).resolve()
    if int(triangle_chunk_size) < 1:
        raise ValueError("triangle_chunk_size must be positive")
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    center_sum = np.zeros(3, dtype=np.float64)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        header = stream.read(84)
        if len(header) != 84:
            raise ValueError(f"Truncated binary STL header: {path}")
        n_triangles = struct.unpack("<I", header[80:84])[0]
        expected_size = 84 + 50 * n_triangles
        if not n_triangles or before.st_size != expected_size:
            raise ValueError(f"Expected a nonempty binary STL of {expected_size} bytes: {path}")
        digest.update(header)
        remaining = n_triangles
        while remaining:
            count = min(int(triangle_chunk_size), remaining)
            raw = stream.read(50 * count)
            if len(raw) != 50 * count:
                raise ValueError(f"Truncated binary STL triangles: {path}")
            digest.update(raw)
            triangles = np.frombuffer(raw, dtype=_STL_TRIANGLE, count=count)
            vertices = triangles["vertices"]
            if not np.isfinite(vertices).all():
                raise ValueError(f"Nonfinite STL coordinates: {path}")
            minimum = np.minimum(minimum, vertices.min(axis=(0, 1)))
            maximum = np.maximum(maximum, vertices.max(axis=(0, 1)))
            center_sum += vertices.astype(np.float64).mean(axis=1).sum(axis=0)
            remaining -= count
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"STL changed while it was being audited: {path}")
    return {
        "source_path": str(path), "source_size_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns, "source_sha256": digest.hexdigest(),
        "n_triangles": n_triangles, "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(), "extent": (maximum - minimum).tolist(),
        "mean_stl_triangle_center_float64": (center_sum / n_triangles).tolist(),
    }


def build_geometry_reference(case_summaries, *, splits_path, splits_sha256, dataset_root):
    """Build the fixed geometry-only protocol from one STL summary per case."""
    summaries = [dict(row) for row in case_summaries]
    if not summaries:
        raise ValueError("At least one STL case is required")
    summaries.sort(key=lambda row: int(row["case_id"]))
    minimum = np.min([row["bbox_min"] for row in summaries], axis=0)
    maximum = np.max([row["bbox_max"] for row in summaries], axis=0)
    reference = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "scope": "all_available_cases_geometry_only",
        "n_cases": len(summaries), "case_ids": [str(row["case_id"]) for row in summaries],
        "dataset_root": str(dataset_root), "splits_path": str(splits_path),
        "splits_sha256": str(splits_sha256),
        "centering": CENTERING, "scaling": SCALING,
        "coordinate_frame": "source_stl_coordinates_as_stored",
        "reference_scale": (maximum - minimum).tolist(),
        "global_bbox_min": minimum.tolist(), "global_bbox_max": maximum.tolist(),
        "max_case_extent": np.max([row["extent"] for row in summaries], axis=0).tolist(),
        "stored_surface": "as_stored_half_body_and_supports_no_reflection_or_doubling",
        "input_scalars": ["inlet_velocity"], "scalar_feature_dim": 1,
        "velocity_scaling": "raw_positive_case_velocity",
        "geometry_transform": "(coordinates - mean(all_case_stl_triangle_centers, axis=0)) / reference_scale",
        "target_transform": "unchanged_native_pressure_and_three_wall_shear_channels_then_train_only_zscore",
        "scope_note": "All case geometries are included by explicit user instruction; no validation/test field targets enter this reference scale.",
        "cases": summaries,
    }
    return validate_geometry_reference(reference)


def validate_geometry_reference(reference, *, expected_case_ids=None):
    """Validate protocol meaning, dimensions, case coverage, and stored bounds.

    This validates the saved audit without reopening its source files. Every STL
    SHA-256 and file size remains available in ``cases`` for provenance checks.
    """
    if not isinstance(reference, dict):
        raise TypeError("geometry reference must be a dictionary")
    if reference.get("schema_version") != SCHEMA_VERSION or reference.get("status") != "complete":
        raise ValueError("A completed geometry reference with schema_version=1 is required")
    if reference.get("centering") != CENTERING or reference.get("scaling") != SCALING:
        raise ValueError("Geometry centering/scaling convention does not match this protocol")
    if reference.get("coordinate_frame") != "source_stl_coordinates_as_stored":
        raise ValueError("Global bounds must use STL coordinates as stored")
    if reference.get("stored_surface") != "as_stored_half_body_and_supports_no_reflection_or_doubling":
        raise ValueError("Reflected, doubled, or reconstructed geometry is incompatible")
    if reference.get("input_scalars") != ["inlet_velocity"] or reference.get("scalar_feature_dim") != 1:
        raise ValueError("Exactly one scalar input, inlet velocity, is required")
    if reference.get("velocity_scaling") != "raw_positive_case_velocity":
        raise ValueError("Velocity must retain the saved raw-scalar convention")
    arrays = {key: np.asarray(reference[key], dtype=np.float64)
              for key in ("reference_scale", "global_bbox_min", "global_bbox_max", "max_case_extent")}
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Every geometry bound/scale must contain three finite values")
    if np.any(arrays["reference_scale"] <= 0) or np.any(arrays["max_case_extent"] <= 0):
        raise ValueError("All coordinate side lengths must be strictly positive")
    if not np.allclose(arrays["reference_scale"], arrays["global_bbox_max"] - arrays["global_bbox_min"], rtol=1e-10, atol=1e-12):
        raise ValueError("reference_scale must equal the global bounding-box full side lengths")
    rows = reference.get("cases", [])
    ids = [str(row["case_id"]) for row in rows]
    if len(rows) != reference.get("n_cases") or not ids or len(ids) != len(set(ids)):
        raise ValueError("One unique STL summary is required per case")
    if ids != list(reference.get("case_ids", [])):
        raise ValueError("case_ids must agree with the ordered case summaries")
    if expected_case_ids is not None and set(ids) != {str(case_id) for case_id in expected_case_ids}:
        raise ValueError("Geometry reference case coverage differs from the requested cases")
    minimum = np.min([row["bbox_min"] for row in rows], axis=0)
    maximum = np.max([row["bbox_max"] for row in rows], axis=0)
    maximum_extent = np.max([row["extent"] for row in rows], axis=0)
    for actual, saved in [(minimum, arrays["global_bbox_min"]), (maximum, arrays["global_bbox_max"]),
                          (maximum_extent, arrays["max_case_extent"])]:
        if actual.shape != (3,) or not np.isfinite(actual).all() or not np.allclose(actual, saved, rtol=1e-10, atol=1e-12):
            raise ValueError("Aggregate geometry bounds disagree with the case summaries")
    for row in rows:
        extent = np.asarray(row["extent"], dtype=np.float64)
        low, high = np.asarray(row["bbox_min"]), np.asarray(row["bbox_max"])
        if extent.shape != (3,) or np.any(extent <= 0) or not np.allclose(extent, high-low, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Invalid per-case bounds for case {row['case_id']}")
        if not isinstance(row.get("source_sha256"), str) or len(row["source_sha256"]) != 64:
            raise ValueError("Every STL requires a SHA-256 provenance entry")
        try:
            int(row["source_sha256"], 16)
        except ValueError:
            raise ValueError("STL SHA-256 must be hexadecimal")
        if int(row.get("n_triangles", 0)) <= 0 or row.get("source_size_bytes") != 84 + 50 * int(row["n_triangles"]):
            raise ValueError("Binary STL triangle count/file size mismatch")
    return reference


def load_geometry_reference(path, *, expected_case_ids=None):
    """Load and validate a saved fixed reference; no source STL is reread."""
    reference = json.loads(Path(path).read_text())
    return validate_geometry_reference(reference, expected_case_ids=expected_case_ids)


def locate_geometry_reference(start_path=None):
    """Locate a workspace/bundled protocol, honoring an explicit path first.

    An invalid ``SURFACE_GEOMETRY_REFERENCE`` is returned unchanged so loading
    fails visibly; a different protocol is never silently substituted. Without
    an existing file, the expected workspace path is returned for a clear error.
    """
    configured = os.environ.get("SURFACE_GEOMETRY_REFERENCE")
    if configured:
        return Path(configured).expanduser()
    start = Path.cwd() if start_path is None else Path(start_path).expanduser()
    if start.is_file():
        start = start.parent
    start = start.resolve()
    candidates = [start / "geometry_reference.json"]
    external = os.environ.get("FULL_DATA_UQ_ROOT")
    if external:
        candidates.append(Path(external).expanduser()/"geometry_reference.json")
    for path in candidates:
        if path.is_file():
            return path
    return start / "geometry_reference.json"


def geometry_scale(reference, *, dtype=np.float32):
    """Return the three shared full side lengths, ready for a model tensor."""
    validate_geometry_reference(reference)
    return np.asarray(reference["reference_scale"], dtype=dtype)


def normalize_geometry(points, stl_centers, reference):
    """Center one case on all stored STL triangle centers and divide by the scale.

    Float32 arithmetic matches the model's coordinate preprocessing. Points may
    be full surface coordinates or STL vertices; normals should not be passed.
    """
    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(stl_centers, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("points must be a finite nonempty (N,3) coordinate array")
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("stl_centers must be all finite nonempty (K,3) triangle centers")
    center = centers.mean(axis=0, dtype=np.float32)
    return (points - center) / geometry_scale(reference)


def geometry_features(stl_centers, surface_points, reference):
    """Return normalized (N,3) positions; concatenate unit normals separately."""
    return normalize_geometry(surface_points, stl_centers, reference)


def velocity_conditioning(inlet_velocity, *, n_points=None):
    """Construct a raw velocity-only scalar of shape (1,1,1) or (1,N,1).

    GeoTransolver uses one global scalar. Transolver broadcasts the same scalar
    to every sampled point. Density remains available for physical postprocessing
    but is intentionally absent from the model-conditioning tensor.
    """
    value = np.asarray(inlet_velocity)
    if value.ndim != 0 or not np.isfinite(value) or float(value) <= 0:
        raise ValueError("inlet_velocity must be one finite strictly positive scalar")
    if n_points is None:
        count = 1
    elif isinstance(n_points, (int, np.integer)) and not isinstance(n_points, bool) and int(n_points) > 0:
        count = int(n_points)
    else:
        raise ValueError("n_points must be a strictly positive integer")
    return np.full((1, count, 1), float(value), dtype=np.float32)


# ===========================================================================
# Introductory geometry and attention figures
# ===========================================================================

"""Plotting helpers for the preprocessing and attention teaching notebooks.

The numerical examples stay in the notebooks. These functions take the already
computed arrays, create a labeled figure, display it, and return the Figure so
readers can save or further customize it. PyVista is imported only when reading
an actual VTP surface; NumPy-only attention examples remain usable without it.
"""





def _finish_teaching_figure(fig, output_path=None):
    """Apply layout once, optionally save, display, and return the figure."""
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()
    return fig


def _read_vtp_arrays(vtp_path):
    """Read original VTP connectivity without reconstructing a point-cloud hull."""
    import pyvista as pv

    mesh = pv.read(vtp_path)
    if "p" not in mesh.point_data and "p" in mesh.cell_data:
        mesh = mesh.cell_data_to_point_data()
    mesh = mesh.triangulate()
    if not mesh.n_faces_strict:
        raise ValueError(f"No surface faces in {vtp_path}")
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return mesh, np.asarray(mesh.points), triangles


def plot_body_fields(vtp_path, *, field_units="kinematic", output_path="ahmed_body_fields.png"):
    """Render raw pressure, shear magnitude, and height on the original mesh.

    Every original polygon is colored by the mean of its vertex values. The
    pressure display uses its 5–95 percentile range, shear uses a 99th-percentile
    cap, and height uses the full range. These are display settings only.
    ``field_units`` is retained for notebook compatibility; no field conversion
    is made here, and the source file does not verify pressure or shear units.
    """
    import pyvista as pv
    path = Path(vtp_path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    mesh = pv.read(path)
    points = np.asarray(mesh.points)
    if "p" not in mesh.point_data or "wallShearStress" not in mesh.point_data:
        raise ValueError("Original point pressure and wall shear arrays are required")
    packed_faces = np.asarray(mesh.faces)
    cells, offset_values, position = [], [], 0
    while position < len(packed_faces):
        count = int(packed_faces[position])
        if count < 3 or position+count >= len(packed_faces):
            raise ValueError("Malformed original VTP polygon connectivity")
        cells.append(packed_faces[position+1:position+count+1])
        offset_values.append((offset_values[-1] if offset_values else 0)+count)
        position += count+1
    connectivity = np.concatenate(cells)
    offsets = np.asarray(offset_values, dtype=np.int64)
    polygons, starts, counts, connectivity = _surface_polygons(points, connectivity, offsets)
    pressure = np.asarray(mesh.point_data["p"]).reshape(-1)
    shear = np.linalg.norm(np.asarray(mesh.point_data["wallShearStress"]), axis=-1)
    height = points[:,2]
    limits = [tuple(np.percentile(pressure, (5,95))),
              (0., max(float(np.percentile(shear,99)), 1e-9)),
              (float(height.min()), float(height.max()))]
    titles = ["CFD pressure", "Wall-shear magnitude", "Geometry: height"]
    cmaps, labels = ["turbo", "plasma", "viridis"], ["Pressure [native units]", "Shear magnitude [native units]", "Height [m]"]
    panels = [_surface_panel(points, polygons, starts, counts, connectivity, values,
                            titles[index], limits[index], cmaps[index], labels[index])
              for index, values in enumerate((pressure, shear, height))]
    fig, audit = _render_polygon_panels(panels, shape=(1,3),
        title="Read the CFD fields before training",
        caption=f"Ahmed body · {path.stem} · Original surface polygons",
        footer="Pressure: 5–95 percentile range. Shear: 99th percentile cap. Height: full range.\n"
               "Arrow ends mark saturated colors. Pressure/shear units are unverified; original values are unchanged.",
        output_path=output_path, show=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha:
        raise ValueError("The source VTP changed during rendering")
    audit.update(source_vtp=str(path.resolve()), source_vtp_sha256=source_sha,
        coordinates_sha256=_array_sha256(points), original_connectivity_sha256=_array_sha256(connectivity),
        original_offsets_sha256=_array_sha256(offsets), source_units_verified=False,
        source_field_units_assumption=field_units, source_file_unchanged=True)
    fig.surface_render_audit = audit
    return fig


def plot_geometry_comparison(vtp_files, titles=None, *, output_path="ahmed_body_comparison.png"):
    """Compare seeded point samples in top/side projections; no field is colored."""
    import pyvista as pv

    titles = titles or [Path(path).stem for path in vtp_files]
    if not vtp_files or len(titles) != len(vtp_files):
        raise ValueError("Provide at least one VTP and one title per VTP")
    fig, axes = plt.subplots(2, len(vtp_files), figsize=(4 * len(vtp_files), 6), squeeze=False)
    for i, (path, title) in enumerate(zip(vtp_files, titles)):
        mesh = pv.read(path)
        idx = np.random.default_rng(i).choice(mesh.n_points, min(10000, mesh.n_points), replace=False)
        points = np.asarray(mesh.points)[idx]
        for row, axis, label in [(0, 1, "Top: y (m)"), (1, 2, "Side: z (m)")]:
            axes[row, i].scatter(points[:, 0], points[:, axis], s=0.3, rasterized=True)
            axes[row, i].set(title=title, xlabel="x (m)", ylabel=label)
            axes[row, i].set_aspect("equal")
        print(f"{title}: {mesh.n_points:,} points; area={mesh.area:.3f} m²")
    return _finish_teaching_figure(fig, output_path)


def plot_field_distributions(fields, channel_names, n_cases, *, output_path="field_distributions.png"):
    """Plot point-weighted channel histograms with sample mean and ±1σ markers."""
    fields = np.asarray(fields)
    if fields.ndim != 2 or fields.shape[1] != len(channel_names):
        raise ValueError("fields must have one column per channel name")
    fig, axes = plt.subplots(1, len(channel_names), figsize=(4 * len(channel_names), 4), squeeze=False)
    for j, (ax, name) in enumerate(zip(axes[0], channel_names)):
        values = fields[:, j]
        ax.hist(values, bins=80, color="steelblue", edgecolor="none", log=True)
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"mean={values.mean():.3g}")
        ax.axvline(values.mean() + values.std(), color="orange", linestyle=":", label="±σ")
        ax.axvline(values.mean() - values.std(), color="orange", linestyle=":")
        ax.set(title=name, xlabel="Value", ylabel="Count (log)")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.legend(fontsize=8)
    fig.suptitle(f"Field distributions (sampled from {n_cases} training files)")
    return _finish_teaching_figure(fig, output_path)


def plot_attention_comparison(random_weights, constructed_weights, labels):
    """Compare two already-computed attention matrices using one color range."""
    vmax = max(np.max(random_weights), np.max(constructed_weights))
    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, weights) in zip(axes, [
        ("Random init — nearly uniform", random_weights),
        ("Hand-written illustrative pattern", constructed_weights),
    ]):
        im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set(xlabel="Key: source of influence", ylabel="Query: sensing point", title=title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                        color="white" if weights[i, j] < 0.55 * vmax else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="Attention weight")
    fig.suptitle(f"Self-attention: random initialization and constructed example (N={n} regions)")
    return _finish_teaching_figure(fig)


def plot_pressure_snapshot(points, pressure, data_label, sample_indices, *, source_vtp=None):
    """Color original CFD polygons, or explicitly plot points for a toy.

    ``sample_indices`` controls only the no-connectivity toy view. Actual CFD
    rendering reads the VTP faces and pressure in their original common order.
    The 5th/95th percentile range emphasizes spatial detail without deleting
    extreme field values from the notebook's subsequent statistics.
    """
    limits = [tuple(np.percentile(pressure, [5, 95]))]
    if source_vtp is not None:

        mesh, vertices, triangles = _read_vtp_arrays(source_vtp)
        values = np.asarray(mesh.point_data["p"]).reshape(-1)
        limits = tuple(np.percentile(values, [5, 95]))
        polygons, starts, counts, connectivity = _surface_polygons(
            vertices, triangles.ravel(), np.arange(1, len(triangles)+1)*3)
        panel = _surface_panel(vertices, polygons, starts, counts, connectivity,
                               values, data_label, limits, "turbo", "Pressure [native units]")
        fig, _ = _render_polygon_panels([panel], shape=(1, 1), title=data_label,
            caption="Original surface polygons; colors average their vertex values",
            footer="Pressure colors use the 5–95 percentile range. Arrow ends mark saturation; values are unchanged.",
            show=True)
        return fig
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    idx = np.asarray(sample_indices)
    sc = ax.scatter(*points[idx].T, c=pressure[idx], cmap="turbo", s=2,
                    vmin=limits[0][0], vmax=limits[0][1])
    fig.colorbar(sc, ax=ax, label="Synthetic scalar", shrink=0.55, pad=0.1, extend="both")
    fig.text(.5, .015, "Colors use the 5–95 percentile range; values outside it use the endpoint colors.",
             ha="center", fontsize=10)
    ax.set(xlabel="x (mesh units)", ylabel="y (mesh units)", zlabel="z (mesh units)", title=data_label)
    ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
    ax.view_init(elev=20, azim=-60)
    return _finish_teaching_figure(fig)


def plot_spatial_statistics(bin_centers, mean_similarity, mean_semivariance, counts, bin_widths, data_label):
    """Display precomputed signed spatial products, semivariance, and pair support."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(bin_centers, mean_similarity, "o-", ms=3)
    axes[0].axhline(0, color="gray", lw=1)
    axes[0].set(ylabel="Mean centered pressure product", title="Signed spatial product")
    axes[1].plot(bin_centers, mean_semivariance, "o-", ms=3)
    axes[1].set(ylabel="Standardized semivariance", title="Pressure differences")
    axes[2].bar(bin_centers, counts, width=bin_widths, align="center")
    axes[2].set(ylabel="Number of sampled pairs", title="Sampling support")
    for ax in axes:
        ax.set_xlabel("Euclidean separation (mesh units)")
        ax.grid(True, alpha=0.25)
    fig.suptitle(data_label, fontsize=10)
    return _finish_teaching_figure(fig)


def plot_spatial_clusters(points, cluster_ids, data_label, n_slices):
    """Plot categorical coordinate-cluster labels; this is not learned slicing."""
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*np.asarray(points).T, c=cluster_ids, cmap="turbo", s=2, alpha=0.75)
    ax.set(xlabel="x (mesh units)", ylabel="y (mesh units)", zlabel="z (mesh units)",
           title=f"{data_label}\nM={n_slices} spatial clusters (KMeans proxy)\n"
                 "Each color is one slice; real slices are learned from physics features.")
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(labelsize=7)
    return _finish_teaching_figure(fig)


def plot_slice_assignments(initial_weights, constructed_weights, regime_order):
    """Show already-computed soft memberships and maximum-weight distributions."""
    n, m = initial_weights.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for col, (tag, weights) in enumerate([
        ("At initialization", initial_weights), ("Constructed from known centroids", constructed_weights),
    ]):
        ax = axes[0, col]
        im = ax.imshow(weights[regime_order].T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set(xlabel=f"Mesh point (sorted by planted regime, N={n})",
               ylabel=f"Slice index (M={m})", title=f"Slice weights w — {tag}")
        fig.colorbar(im, ax=ax, label="w[n, m]")
        ax = axes[1, col]
        ax.hist(weights.max(1), bins=40, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(1 / m, color="red", ls="--", lw=2, label=f"uniform 1/M = {1 / m:.3f}")
        ax.axvline(weights.max(1).mean(), color="orange", ls="--", lw=2,
                   label=f"mean = {weights.max(1).mean():.3f}")
        ax.set(xlabel="max slice weight per point", ylabel="count", title=f"Point commitment — {tag}", xlim=(0, 1))
        ax.legend(fontsize=8)
    return _finish_teaching_figure(fig)


def plot_token_features(tokens):
    """Plot the M×C token matrix; inputs are not recomputed in this helper."""
    m, c = tokens.shape
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(tokens, aspect="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label="Feature value")
    ax.set(xlabel=f"Channel index (C={c})", ylabel=f"Slice / token index (M={m})",
           title="Step 2: physics-aware tokens z (shape M × C)\nEach row summarizes synthetic point features")
    return _finish_teaching_figure(fig)


def plot_token_attention(attention_weights, n_points):
    """Plot the token attention matrix with numeric entries and its mesh-scale contrast."""
    m = attention_weights.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(attention_weights, cmap="viridis", vmin=0)
    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{attention_weights[i, j]:.2f}", ha="center", va="center",
                    color="white" if attention_weights[i, j] < 0.15 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Attention weight")
    ax.set(xlabel="Key token (slice index)", ylabel="Query token (slice index)",
           title=f"Step 3: token attention ({m}×{m})\n"
                 f"Replaces a {n_points}×{n_points} = {n_points**2:,}-entry matrix")
    return _finish_teaching_figure(fig)


def configure_tutorial_style():
    """Set presentation defaults used by Notebook 4's coordinate analogy."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def plot_geometry_cloud(geometry, label):
    """Orient the coordinate toy with a point-cloud view, colored by original z."""
    points = np.asarray(geometry)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(*points.T, c=points[:, 2], cmap="plasma", s=6, alpha=0.8)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="z (m)")
    ax.set(title=label, xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
    ax.view_init(25, -55)
    return _finish_teaching_figure(fig)


def plot_coordinate_snapshots(geometry, standard_history, anchored_history,
                              mse_standard, mse_anchored, std_standard, std_anchored,
                              snapshot_layers, noise_scale, context_strength, *, padding=0.45):
    """Compare paired toy trajectories under shared axes and original-height colors."""
    points = np.asarray(geometry)
    vmin, vmax = points[:, 2].min(), points[:, 2].max()
    limits = [(points[:, i].min() - padding, points[:, i].max() + padding) for i in range(3)]
    fig, axes = plt.subplots(2, len(snapshot_layers), figsize=(5 * len(snapshot_layers), 9),
                             subplot_kw={"projection": "3d"}, squeeze=False)
    rows = [(standard_history, "Unanchored\ncoordinate toy", "#d62728", mse_standard, std_standard),
            (anchored_history, "Anchored\ncoordinate toy", "#2ca02c", mse_anchored, std_anchored)]
    for col, layer in enumerate(snapshot_layers):
        for row, (history, label, color, mse, std) in enumerate(rows):
            ax = axes[row, col]
            state = np.asarray(history[layer])
            ax.scatter(*state.T, c=points[:, 2], cmap="plasma", s=8, alpha=0.75, vmin=vmin, vmax=vmax)
            ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_zlim(limits[2])
            ax.view_init(25, -55); ax.axis("off")
            ax.set_title(f"Layer {layer}\nMSE={mse[layer]:.4f}  σ={std[layer]:.4f}", fontsize=11, color=color)
            if col == 0:
                ax.text2D(-0.10, 0.50, label, transform=ax.transAxes, fontsize=12,
                          fontweight="bold", color=color, rotation=90, va="center")
    fig.suptitle(f"Coordinate-restoration analogy (noise σ={noise_scale}, context α={context_strength})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.95])
    plt.show()
    return fig


def plot_coordinate_drift(layers, mse_standard, mse_anchored, std_standard, std_anchored):
    """Display descriptive coordinate-toy errors; shaded gaps are not intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, a, b, ylabel, title in [
        (axes[0], mse_standard, mse_anchored, "MSE from original geometry", "MSE vs depth"),
        (axes[1], std_standard, std_anchored, "Std-dev of point displacement", "Point-cloud fuzziness vs depth"),
    ]:
        ax.plot(layers, a, "o-", color="#d62728", lw=2.5, ms=4, label="Unanchored toy")
        ax.plot(layers, b, "s-", color="#2ca02c", lw=2.5, ms=4, label="Anchored toy")
        ax.fill_between(layers, b, a, alpha=0.18, color="#d62728")
        ax.set(xlabel="Layer depth", ylabel=ylabel, title=title)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ratio = mse_standard[-1] / (mse_anchored[-1] + 1e-9)
    axes[0].text(0.97, 0.97, f"Toy MSE ratio: {ratio:.0f}×\nat layer {layers[-1]}",
                 transform=axes[0].transAxes, fontsize=11, ha="right", va="top", fontweight="bold",
                 bbox=dict(facecolor="#e8f5e9", alpha=0.9, edgecolor="#2ca02c"))
    fig.suptitle("Coordinate-restoration error: a toy comparison", fontsize=14, fontweight="bold")
    return _finish_teaching_figure(fig)


def plot_context_sensitivity(layers, alpha_results, mse_standard, std_standard):
    """Render the paired restoring-strength experiment; return both figures."""
    alphas = list(alpha_results)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(alphas)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, reference, key, ylabel, title in [
        (axes[0], mse_standard, "mse", "MSE from original", "MSE vs depth — effect of α"),
        (axes[1], std_standard, "noise", "Std-dev of displacement", "Fuzziness vs depth — effect of α"),
    ]:
        ax.plot(layers, reference, "--", color="gray", lw=2, label="Unanchored (α=0)", zorder=0)
        for alpha, color in zip(alphas, colors):
            ax.plot(layers, alpha_results[alpha][key], "o-", color=color, lw=2, ms=3,
                    label=f"Anchored toy α={alpha}")
        ax.set(xlabel="Layer depth", ylabel=ylabel, title=title)
        ax.legend(fontsize=10, loc="upper left"); ax.grid(True, alpha=0.3)
    fig.suptitle("Toy restoring-strength sensitivity (paired weights and noise)", fontsize=14, fontweight="bold")
    _finish_teaching_figure(fig)
    final_fig, ax = plt.subplots(figsize=(6, 4))
    final_mse = [alpha_results[alpha]["mse"][-1] for alpha in alphas]
    ax.axhline(mse_standard[-1], color="gray", lw=2, ls="--", label="Unanchored toy")
    ax.plot(alphas, final_mse, "o-", color="#2ca02c", lw=2.5, ms=9, label="Anchored toy")
    for alpha, value, color in zip(alphas, final_mse, colors):
        ax.annotate(f"α={alpha}", xy=(alpha, value), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, color=color)
    ax.set(xlabel="Context strength α", ylabel=f"Final MSE (layer {layers[-1]})",
           title="Final drift vs context strength α")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    _finish_teaching_figure(final_fig)
    return fig, final_fig


def plot_gate_coefficients(weights):
    """Plot saved scalar self/cross coefficients; this does not measure influence."""
    weights = np.asarray(weights)
    fig, ax = plt.subplots(figsize=(10, 4))
    depth = np.arange(len(weights))
    ax.plot(depth, weights, "o-", label="w: self-attention coefficient")
    ax.plot(depth, 1 - weights, "s-", label="1-w: cross-attention coefficient")
    ax.axhline(0.5, color="gray", ls="--", label="initial coefficient")
    ax.set(xlabel="GALE block index", ylabel="Mixing coefficient", ylim=(0, 1),
           title="Saved scalar gate coefficients (one per block)")
    ax.legend(); ax.grid(True, alpha=0.25)
    return _finish_teaching_figure(fig)


def plot_surface_data_contract(reference, *, output_path=None):
    """Diagram the saved geometry/speed contract and the separate target paths.

    The case count and scale are read from the audited manifest. Arrows describe
    preprocessing dependencies, not CFD causality or measured model accuracy.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    validate_geometry_reference(reference)
    scale = ", ".join(f"{value:.6f}" for value in reference["reference_scale"])
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")

    def box(x, y, width, height, title, body, color):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.009",
                               facecolor=color, edgecolor="#566573", linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height - .035, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color="#17202A")
        ax.text(x + width / 2, y + height / 2 - .025, body, ha="center", va="center",
                fontsize=11, linespacing=1.45, color="#17202A")

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                    linewidth=1.5, color="#566573"))

    box(.01, .66, .22, .23, "Case geometry", "Surface positions and normals\nSTL vertices and triangle centers\nActual meshes, not tabulated dimensions", "#EBF5FB")
    box(.28, .66, .30, .23, "One frozen coordinate reference",
        f"Union box of all {reference['n_cases']} STLs\ns = [{scale}]\nCase center = mean(all STL triangle centers)\nNormalized coordinates = (x − case center) / s", "#E8F8F5")
    box(.65, .66, .33, .23, "Model inputs: geometry + one scalar",
        "Transolver speed: (1, N, 1)\nGeoTransolver speed: (1, 1, 1)\nLocal position/normal channels: 6\nGeoTransolver also receives STL context", "#E8F8F5")
    box(.01, .33, .22, .21, "Case info → Zarr",
        "stream_velocity = this case's U\nDensity retained for provenance\nand conditional physical postprocessing", "#FEF9E7")
    box(.28, .33, .30, .21, "Select the scalar speed",
        "External datapipe: [density, velocity]\nVelocityOnlyDataPipe: [velocity]\nNo density feature or reference division", "#FEF9E7")
    box(.65, .33, .33, .21, "Four predicted surface channels",
        "Pressure and wall shear x/y/z\nPredictions keep the input point order\nCFD targets are not model inputs", "#EBF5FB")
    box(.01, .03, .57, .19, "Separate target lane: choose the format before training",
        "Production: native fields → 458-case training mean/std\n"
        "Teaching conversion: fields / per-case q → generated-training mean/std", "#F4ECF7")
    box(.65, .03, .33, .19, "Loss and inverse transformation",
        "Compare predictions with normalized targets\nUndo the saved z-score once\nDo not add U² to native-field predictions", "#F4ECF7")
    arrow((.24, .775), (.267, .775)); arrow((.59, .775), (.637, .775))
    arrow((.24, .435), (.267, .435)); arrow((.59, .46), (.637, .69))
    arrow((.815, .645), (.815, .555)); arrow((.815, .315), (.815, .235))
    arrow((.59, .125), (.637, .125))
    fig.suptitle("Velocity is the only scalar condition; geometry and target provenance remain explicit", fontsize=15)
    return _finish_teaching_figure(fig, output_path)


# ===========================================================================
# CFD surface rendering, diagnostics, and notebook support
# ===========================================================================

"""Plotting and surface diagnostics for Transolver notebooks 3, 5 and 6.

Scientific calculations remain visible in notebooks or named reusable functions.
Mesh plots use original VTP connectivity; no Delaunay reconstruction or point
subsampling is used for full-surface prediction maps. Imports of PyVista/Torch
are lazy so ordinary tables and learning curves also work on CPU-only systems.
"""


def _finish(fig, output_path=None, show=True):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _mesh_screenshot(points, triangles, values, cmap, limits, label):
    import pyvista as pv
    points, triangles = np.asarray(points), np.asarray(triangles)
    values = np.asarray(values)
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Expected points (N,3) and original-mesh triangles (K,3)")
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("A finite point scalar is required for every mesh vertex")
    if triangles.size == 0 or triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("Triangle connectivity is empty or references missing vertices")
    faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
    surface = pv.PolyData(points, faces)
    surface.point_data["value"] = values
    plotter = pv.Plotter(off_screen=True, window_size=(850, 620))
    try:
        plotter.set_background("white")
        plotter.add_mesh(surface, scalars="value", cmap=cmap, clim=limits,
                         interpolate_before_map=True, show_edges=False,
                         ambient=0.65, diffuse=0.35,
                         scalar_bar_args={"title": label, "vertical": False,
                                          "position_x": 0.15, "width": 0.7,
                                          "position_y": 0.03, "height": 0.07,
                                          "title_font_size": 15, "label_font_size": 13,
                                          "color": "black", "fmt": "%.3g"})
        center = (points.min(0) + points.max(0)) / 2
        length = max(float(np.ptp(points, axis=0).max()), 1e-6)
        plotter.camera_position = [center + length*np.array([1.7, 2.2, 1.2]), center, [0,0,1]]
        plotter.camera.parallel_projection = True
        plotter.reset_camera()
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


def render_mesh_panels(points, triangles, scalar_arrays, titles, *, cmaps=None,
                       limits=None, labels=None, ncols=3, figsize=None,
                       output_path=None, show=True):
    """Render continuous vertex colors on the supplied physical mesh.

    Triangle interpolation only changes the displayed colors. It adds no CFD
    samples and does not enter force integration or reported pointwise metrics.
    """
    count = len(scalar_arrays)
    if not count or len(titles) != count:
        raise ValueError("Provide one title per scalar array")
    cmaps = cmaps or ["viridis"]*count
    limits = limits or [None]*count
    labels = labels or ["Stored field units"]*count
    nrows = (count+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5*ncols, 3.9*nrows), squeeze=False)
    for index, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if index < count:
            ax.imshow(_mesh_screenshot(points, triangles, scalar_arrays[index],
                                      cmaps[index], limits[index], labels[index]))
            ax.set_title(titles[index], fontsize=10, pad=13)
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.04, hspace=.28, wspace=.05)
    return _finish(fig, output_path, show)


def find_ahmed_vtp(case_id, dataset_root=None):
    """Locate original topology; portable override: AHMED_DATASET_ROOT."""
    roots = []
    if dataset_root is not None:
        roots.append(Path(dataset_root))
    if os.environ.get("AHMED_DATASET_ROOT"):
        roots.append(Path(os.environ["AHMED_DATASET_ROOT"]))
    for ancestor in [Path.cwd(), *Path.cwd().parents]:
        roots.append(ancestor / "physicsnemo_ahmed_body_dataset_vv1")
    roots.append(Path(__file__).resolve().parent / "data/ahmed_body")
    for root in roots:
        for suffix in ("dataset/reformatted_dataset", "reformatted_dataset"):
            path = root / suffix / f"run_{case_id}" / f"surface_{case_id}.vtp"
            if path.is_file():
                return path
    raise FileNotFoundError(f"Original case {case_id} VTP not found; set AHMED_DATASET_ROOT")


def original_surface_mesh(case, dataset_root=None):
    """Verify coordinate order before attaching predictions to original faces."""
    import pyvista as pv
    path = Path(case.get("mesh_path") or find_ahmed_vtp(case["case_id"], dataset_root))
    mesh = pv.read(path)
    xyz = np.asarray(case["coordinates"], dtype=np.float64)
    source = np.asarray(mesh.points, dtype=np.float64)
    if source.shape != xyz.shape:
        raise ValueError(f"{path}: full surface expected ({source.shape} != {xyz.shape}); no subset interpolation")
    max_error = float(np.max(np.abs(source-xyz)))
    tolerance = max(float(np.ptp(source,axis=0).max())*2e-6, 1e-7)
    if max_error > tolerance:
        raise ValueError(f"{path}: saved coordinate order/units mismatch, maximum error {max_error:g} > {tolerance:g}")
    triangles = np.asarray(mesh.triangulate().faces).reshape(-1,4)[:,1:]
    print(f"case {case['case_id']}: original VTP, {len(xyz):,} vertices, {len(triangles):,} triangles; coordinate-order max error {max_error:.3g} m")
    return xyz, triangles



def export_prediction_vtp(case, output_path, dataset_root=None):
    """Export native vertex fields on the untouched original VTP topology.

    Pressure/shear are already in the checkpoint's native stored convention:
    no velocity-squared factor is applied. Raw CFD cell arrays are preserved.
    Optional flow values describe supplied conditioning metadata, not an
    independently verified CFD-solver boundary condition.
    """
    import pyvista as pv
    source_path=Path(case.get('mesh_path') or find_ahmed_vtp(case['case_id'],dataset_root)).resolve()
    output_path=Path(output_path).expanduser().resolve()
    if output_path==source_path:
        raise ValueError('Export must not overwrite the original CFD VTP')
    xyz,_=original_surface_mesh(case,dataset_root)
    mesh=pv.read(source_path)
    prediction=np.asarray(case['mean'])
    if prediction.shape!=(len(xyz),4) or not np.isfinite(prediction).all():
        raise ValueError('Expected finite native pressure and all three WSS components')
    mesh.point_data['PredictedPressure']=prediction[:,0]
    mesh.point_data['PredictedWallShearStress']=prediction[:,1:4]
    if 'target' in case:
        target=np.asarray(case['target'])
        if target.shape!=prediction.shape or not np.isfinite(target).all():
            raise ValueError('Reference fields must match predicted vertex fields')
        mesh.point_data['ReferencePressure']=target[:,0]
        mesh.point_data['ReferenceWallShearStress']=target[:,1:4]
        mesh.point_data['AbsolutePressureError']=np.abs(prediction[:,0]-target[:,0])
        mesh.point_data['WallShearErrorMagnitude']=np.linalg.norm(prediction[:,1:4]-target[:,1:4],axis=1)
    if 'std' in case:
        spread=np.asarray(case['std'])
        if spread.shape!=prediction.shape or not np.isfinite(spread).all() or np.any(spread<0):
            raise ValueError('Spread must be finite, nonnegative and match all vertex fields')
        mesh.point_data['PredictedPressureStd']=spread[:,0]
        mesh.point_data['PredictedWallShearStressStd']=spread[:,1:4]
    for key,name in [('stream_velocity','ConditioningInletVelocity'),('air_density','ConditioningAirDensity')]:
        if key in case:
            value=float(np.asarray(case[key]).item())
            if not np.isfinite(value) or value<=0:raise ValueError('Positive finite case flow values are required')
            mesh.field_data[name]=np.array([value])
    mesh.field_data['InferenceCaseID']=np.array([str(case['case_id'])])
    mesh.field_data['NativeUnitsStatus']=np.array(['Native stored pressure/WSS; physical units unverified; no U^2 rescaling'])
    mesh.field_data['ConditioningMetadataStatus']=np.array(['Supplied case metadata; not independently verified CFD solver boundary data'])
    output_path.parent.mkdir(parents=True,exist_ok=True)
    mesh.save(output_path,binary=True)
    print('Exported continuous ParaView surface:',output_path)
    return output_path


def _array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _surface_polygons(points, connectivity, offsets):
    """Return only supplied mesh polygons, without inferring any connections."""
    points = np.asarray(points)
    connectivity, offsets = np.asarray(connectivity), np.asarray(offsets)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or connectivity.ndim != 1 or offsets.ndim != 1
            or not np.issubdtype(connectivity.dtype, np.integer)
            or not np.issubdtype(offsets.dtype, np.integer) or not offsets.size):
        raise ValueError("Provide finite points (N,3) and original integer connectivity/offset arrays")
    starts = np.r_[0, offsets[:-1]]
    counts = offsets-starts
    if (np.any(counts < 3) or offsets[-1] != len(connectivity) or not connectivity.size
            or connectivity.min() < 0 or connectivity.max() >= len(points)):
        raise ValueError("Invalid original polygon connectivity or end offsets")
    polygons = [points[connectivity[start:end]] for start, end in zip(starts, offsets)]
    return polygons, starts, counts, connectivity


def _surface_panel(points, polygons, starts, counts, connectivity, values,
                   title, limits, cmap, label):
    values = np.asarray(values)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("Each field must contain one finite value per original vertex")
    limits = tuple(map(float, limits))
    if not np.isfinite(limits).all() or limits[0] > limits[1]:
        raise ValueError("Provide increasing finite color limits")
    if limits[0] == limits[1]:
        limits = (limits[0]-max(abs(limits[0])*1e-6, 1e-9), limits[1]+max(abs(limits[1])*1e-6, 1e-9))
    face_values = np.add.reduceat(values[connectivity].astype(float), starts)/counts
    return dict(points=np.asarray(points), polygons=polygons, values=values,
                face_values=face_values, title=title, limits=limits, cmap=cmap, label=label)


def _render_polygon_panels(panels, *, shape, title, caption, footer,
                           output_path=None, show=True):
    """CPU rendering of every original polygon; averages affect color only."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    shape = tuple(map(int, shape))
    if len(shape) != 2 or min(shape) < 1 or shape[0]*shape[1] < len(panels):
        raise ValueError("The layout must have space for every panel")
    wide = shape[0] == 1
    figsize = (5.75*shape[1], 7.3) if wide else (6.75*shape[1], 6.*shape[0])
    fig = plt.figure(figsize=figsize, facecolor="white")
    records = []
    for index, panel in enumerate(panels):
        points, values, face_values = panel["points"], panel["values"], panel["face_values"]
        low, high = panel["limits"]
        norm = Normalize(vmin=low, vmax=high, clip=True)
        ax = fig.add_subplot(shape[0], shape[1], index+1, projection="3d")
        collection = Poly3DCollection(panel["polygons"],
            facecolors=plt.get_cmap(panel["cmap"])(norm(face_values)),
            edgecolors="none", linewidths=0., antialiased=False, zsort="average")
        ax.add_collection3d(collection)
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        extent = np.maximum(maximum-minimum, 1e-9)
        center = .5*(minimum+maximum)
        for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(center[dim]-.505*extent[dim], center[dim]+.505*extent[dim])
        ax.set_box_aspect(extent)
        ax.view_init(elev=25, azim=45)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_title(panel["title"], fontsize=17 if wide else 18,
                     fontweight="bold", color="#142638", pad=0)
        below, above = float(np.mean(values < low)), float(np.mean(values > high))
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=panel["cmap"]), ax=ax,
            orientation="horizontal", fraction=.045, pad=.005, shrink=.86,
            aspect=24, extend=extend)
        bar.ax.tick_params(labelsize=12)
        bar.locator = MaxNLocator(nbins=4)
        bar.update_ticks()
        bar.set_label(panel["label"], fontsize=13, labelpad=5)
        bar.outline.set_linewidth(.6)
        records.append(dict(title=panel["title"], cmap=panel["cmap"],
            display_range=[low, high], source_point_array_sha256=_array_sha256(values),
            display_face_mean_array_sha256=_array_sha256(face_values),
            original_min=float(values.min()), original_max=float(values.max()),
            fraction_below_range=below, fraction_above_range=above,
            fraction_saturated=below+above, colorbar_extend=extend,
            original_vertices=len(points), original_polygons=len(panel["polygons"])))
    fig.suptitle(title, fontsize=24, fontweight="bold", color="#142638", y=.975)
    fig.text(.5, .915 if wide else .925, caption, ha="center",
             fontsize=12 if wide else 14, color="#44576a")
    fig.text(.5, .035 if wide else .018, footer, ha="center", va="bottom",
             fontsize=11 if wide else 12, color="#344456", linespacing=1.6)
    fig.subplots_adjust(left=.012, right=.988, top=.84 if wide else .895,
        bottom=.17 if wide else .10, wspace=.035, hspace=.12 if wide else .38)
    audit = dict(layout=list(shape), panels=records, footer=footer,
                 rendering="Original polygons colored by mean of original vertex values; no mesh or field subsampling",
                 geometry_unchanged=True, numerical_fields_unchanged=True)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, facecolor="white")
        audit.update(path=str(path.resolve()), png_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if show:
        plt.show()
    return fig, audit


def _surface_color_limits(targets, means, sizes, pressure_percentiles, size_color_quantile):
    pressures = np.concatenate([np.asarray(value) for value in [*targets, *means]])
    if pressure_percentiles is None:
        pressure_limits = (float(pressures.min()), float(pressures.max()))
    else:
        lo, hi = pressure_percentiles
        if not 0 <= lo < hi <= 100:
            raise ValueError("Pressure percentiles must satisfy 0 <= low < high <= 100")
        pressure_limits = tuple(np.percentile(pressures, (lo, hi)))
    if size_color_quantile is not None and not 0 < size_color_quantile <= 1:
        raise ValueError("size_color_quantile must be in (0,1] or None")
    sizes = np.concatenate(sizes)
    cap = float(sizes.max()) if size_color_quantile is None else float(np.quantile(sizes, size_color_quantile))
    return pressure_limits, max(cap, 1e-9)


def _surface_color_footer(pressure_percentiles, size_color_quantile, with_width):
    pressure_text = ("full range" if pressure_percentiles is None else
                     f"{pressure_percentiles[0]:g}–{pressure_percentiles[1]:g} percentile range")
    size_text = "full range" if size_color_quantile is None else f"{100*size_color_quantile:g}th percentile cap"
    quantity = "Error and half-width" if with_width else "Error"
    return (f"Pressure: shared {pressure_text}. {quantity}: shared {size_text}.\n"
            "Arrow ends mark saturated colors; original field values and metrics are unchanged.")


def render_surface_comparison(points, connectivity, offsets, target, mean, *,
                              halfwidth=None, model_label="Model", case_id=None,
                              epoch=None, speed=None, shape=None, output_path=None,
                              show=True, pressure_percentiles=(5, 95), size_color_quantile=.99):
    """Render one archived CFD case and return ``(figure, audit)`` on CPU.

    ``connectivity`` and exclusive ``offsets`` are the original mesh arrays.
    Every polygon is displayed, with the mean of its original vertex values as
    its color. Display percentiles improve contrast without changing fields or
    metrics. Reference and prediction share pressure limits; error and interval
    half-width share a nonnegative limit. Use ``shape=(2,2)`` for a square post.
    """
    points, target, mean = np.asarray(points), np.asarray(target), np.asarray(mean)
    polygons, starts, counts, connectivity = _surface_polygons(points, connectivity, offsets)
    if target.shape != (len(points),) or mean.shape != target.shape:
        raise ValueError("Target and mean must have one pressure value per original vertex")
    error = np.abs(target-mean)
    arrays = [target, mean, error]
    sizes = [error]
    titles = ["CFD reference pressure", f"{model_label} pressure", "Absolute pressure error"]
    labels = ["Pressure [native units]", "Pressure [native units]", "Error [native units]"]
    if halfwidth is not None:
        halfwidth = np.asarray(halfwidth)
        if halfwidth.shape != target.shape or np.any(halfwidth < 0):
            raise ValueError("Half-width must contain one nonnegative value per vertex")
        arrays.append(halfwidth); sizes.append(halfwidth)
        titles.append("90% interval half-width"); labels.append("Half-width [native units]")
    pressure_limits, cap = _surface_color_limits([target], [mean], sizes,
                                                pressure_percentiles, size_color_quantile)
    limits = [pressure_limits, pressure_limits]+[(0., cap)]*(len(arrays)-2)
    cmaps = ["turbo", "turbo", "viridis", "magma"]
    panels = [_surface_panel(points, polygons, starts, counts, connectivity, values,
                            panel_title, limits[index], cmaps[index], labels[index])
              for index, (values, panel_title) in enumerate(zip(arrays, titles))]
    caption = ["Ahmed body", model_label]
    if case_id is not None: caption.append(f"Case {case_id}")
    if epoch is not None: caption.append(f"Frozen epoch {int(epoch)}")
    if speed is not None: caption.append(f"Inlet {float(speed):g} m/s")
    fig, audit = _render_polygon_panels(panels, shape=shape or (1, len(panels)),
        title="From CFD to AI prediction and uncertainty" if halfwidth is not None else "From CFD to AI prediction",
        caption=" · ".join(caption),
        footer=_surface_color_footer(pressure_percentiles, size_color_quantile, halfwidth is not None),
        output_path=output_path, show=show)
    audit.update(pressure_display_percentiles=list(pressure_percentiles) if pressure_percentiles is not None else None,
        size_color_quantile=size_color_quantile, coordinates_sha256=_array_sha256(points),
        original_connectivity_sha256=_array_sha256(connectivity),
        original_offsets_sha256=_array_sha256(np.asarray(offsets)),
        pressure_rmse=float(np.sqrt(np.mean(error.astype(float)**2))),
        pressure_relative_l2=float(np.linalg.norm(error.astype(float))/max(np.linalg.norm(target.astype(float)), 1e-12)),
        full_surface_error_max=float(error.max()),
        calibrated_point_coverage=float(np.mean(error <= halfwidth)) if halfwidth is not None else None)
    return fig, audit


def plot_surface_predictions(cases, *, pressure_multiplier=None, title="Frozen checkpoint",
                             output_path=None, dataset_root=None, show=True,
                             size_color_quantile=.99, point_std_floor=0.,
                             pressure_percentiles=(5, 95)):
    """Original mesh polygons with shared vivid pressure and diagnostic scales.

    Every original vertex enters the printed metrics. Polygon colors average
    their vertex fields; percentile limits affect display only. Set either
    percentile option to ``None`` to use that quantity's full range.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("Provide at least one complete surface case")
    targets, means = [c['target'][:,0] for c in cases], [c['mean'][:,0] for c in cases]
    errors = [np.abs(target-mean) for target, mean in zip(targets, means)]
    widths = ([pressure_multiplier*np.maximum(c['std'][:,0], point_std_floor) for c in cases]
              if pressure_multiplier is not None else [])
    common, cap = _surface_color_limits(targets, means, errors+widths,
                                       pressure_percentiles, size_color_quantile)
    panels, columns = [], 4 if widths else 3
    for row, case in enumerate(cases):
        xyz, triangles = original_surface_mesh(case, dataset_root)
        polygons, starts, counts, connectivity = _surface_polygons(
            xyz, triangles.ravel(), np.arange(1, len(triangles)+1)*3)
        if output_path is not None:
            image_path = Path(output_path)
            export_prediction_vtp(case, image_path.with_name(image_path.stem+f"_case{case['case_id']}.vtp"), dataset_root)
        arrays = [targets[row], means[row], errors[row]]+([widths[row]] if widths else [])
        names = ["CFD pressure", "Predicted pressure", "Absolute error", "90% interval half-width"]
        labels = ["Pressure [native units]", "Pressure [native units]", "Error [native units]", "Half-width [native units]"]
        cmaps = ["turbo", "turbo", "viridis", "magma"]
        for col, values in enumerate(arrays):
            panels.append(_surface_panel(xyz, polygons, starts, counts, connectivity, values,
                f"Case {case['case_id']} · {names[col]}", common if col < 2 else (0., cap), cmaps[col], labels[col]))
        text = (f"Case {case['case_id']}: full-surface pressure RMSE="
                f"{np.sqrt(np.mean(errors[row].astype(float)**2)):.5g}, "
                f"relative L2={np.linalg.norm(errors[row].astype(float))/max(np.linalg.norm(targets[row].astype(float)), 1e-12):.2%}, "
                f"actual maximum error={errors[row].max():.5g} native units; display cap={cap:.5g}")
        if widths:
            text += (f", calibrated point coverage={np.mean(errors[row] <= widths[row]):.3f}, "
                     f"actual maximum half-width={widths[row].max():.5g}")
        print(text)
    fig, audit = _render_polygon_panels(panels, shape=(len(cases), columns), title=title,
        caption="Original surface polygons; colors average their original vertex values",
        footer=_surface_color_footer(pressure_percentiles, size_color_quantile, bool(widths)),
        output_path=output_path, show=show)
    fig.surface_render_audit = audit
    return fig


def plot_measured_training(history, *, selected_epoch=None, continuation=None,
                           title="Measured training history", output_path=None):
    """Full loss history and late-window detail, using actual saved epoch rows."""
    def value(row, train):
        keys=("train_mse","train_loss","training_mse","train") if train else ("validation_mse","val_mse","val_loss","validation")
        return next(float(row[key]) for key in keys if key in row)
    fig, axes=plt.subplots(1,2,figsize=(13,4.3))
    last=int(history[-1]['epoch']); extra=continuation or []
    for ax in axes:
        for train,color,label in [(True,'#2878a5','Training'),(False,'#dc7b36','Validation')]:
            ax.plot([r['epoch'] for r in history],[value(r,train) for r in history],color=color,label=label)
            if extra:
                ax.plot([last]+[r['epoch'] for r in extra],[value(history[-1],train)]+[value(r,train) for r in extra],
                        'o--',color=color,ms=4,label=label+' teaching continuation')
        if selected_epoch is not None:
            ax.axvline(selected_epoch,color='black',ls='--',lw=1,label=f'Frozen evaluated epoch {selected_epoch}')
        if extra: ax.axvline(last+.5,color='gray',ls=':')
        ax.set(xlabel='Completed epoch',ylabel='Four-channel normalized MSE');ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[0].set_title('Full measured history');axes[0].legend(fontsize=7)
    axes[1].set_xlim(max(1,last-50),(extra[-1]['epoch'] if extra else last)+1)
    late=[r for r in history+extra if int(r['epoch'])>=last-50]
    values=[value(r,t) for r in late for t in (True,False)];lo,hi=min(values),max(values);pad=max((hi-lo)*.1,1e-7)
    axes[1].set_ylim(max(0,lo-pad),hi+pad);axes[1].set_title('Last 50 epochs and separate continuation' if extra else 'Last 50 epochs')
    fig.suptitle(title);fig.tight_layout()
    return _finish(fig,output_path)


def capture_slice_weights(model, sample, device, *, geotransolver=False, block_index=0):
    """Capture real logits safely and reproduce native temperature-scaled softmax."""
    import torch
    model.eval(); attn=model.blocks[block_index].Attn; captured={}
    handle=attn.in_project_slice.register_forward_hook(lambda module,args,out:captured.__setitem__('logits',out.detach()))
    try:
        with torch.no_grad():
            embedding=sample['embeddings'].to(device);fx=sample['fx'].to(device)
            if geotransolver:
                model(global_embedding=fx,local_embedding=embedding,
                      geometry=sample['geometry'].to(device),local_positions=embedding[:,:,:3])
            else:
                model(fx=fx,embedding=embedding)
    finally:handle.remove()
    logits=captured['logits'].squeeze(0)
    if logits.dim()==2:logits=logits.unsqueeze(1)
    tau=getattr(attn,'temperature',None)
    if tau is not None:
        tau=torch.clamp(tau.detach(),.5,5).reshape(-1)
        logits=logits/(tau.view(1,-1,1).to(logits.device) if tau.numel()==logits.shape[1] else tau[0].to(logits.device))
    weights=torch.softmax(logits.double(),dim=-1).cpu().numpy()
    return {'heads':weights,'mean':weights.mean(1),'temperature':float(tau.mean()) if tau is not None else None,
            'coordinates':embedding[0,:,:3].cpu().numpy(),'block_index':block_index}


def plot_slice_memberships(routing, *, title="Slice routing diagnostic", output_dir=None, head_index=None):
    """Inspect one head, or explicitly label an index-average diagnostic."""
    if head_index is None:
        A=routing['mean']; weight_label='Head-averaged routing weight'; mass_label='Summed head-average weight'
        maximum_label='Maximum head-average membership'; concentration_title='Index-average concentration'
        subtitle='Sampled model inputs; head slice indices are independently permutable'; mass_kind='index-average'
    else:
        if not 0 <= head_index < routing['heads'].shape[1]:raise ValueError('Routing head index is out of range')
        A=routing['heads'][:,head_index,:]; weight_label=f'Head {head_index} routing weight'; mass_label='Summed within-head weight'
        maximum_label=f'Maximum head {head_index} membership'; concentration_title='Within-head concentration'
        subtitle=f'Sampled model inputs; head {head_index}, with slice labels local to this head'; mass_kind='within-head'
    coords=routing['coordinates'];mass=A.sum(0);top=np.argsort(mass)[-4:][::-1]
    fig=plt.figure(figsize=(13,9))
    for i,slice_id in enumerate(top):
        ax=fig.add_subplot(2,2,i+1,projection='3d');values=A[:,slice_id]
        art=ax.scatter(*coords.T,c=values,s=2,cmap='magma',vmin=values.min(),vmax=values.max())
        fig.colorbar(art,ax=ax,shrink=.5,label=weight_label)
        ax.set_title(f'Slice {slice_id}: {100*mass[slice_id]/mass.sum():.1f}% of {mass_kind} mass')
        ax.set(xlabel='Scaled x',ylabel='Scaled y',zlabel='Scaled z');ax.view_init(20,-55)
    fig.suptitle(title+'\n'+subtitle)
    fig.tight_layout();_finish(fig,Path(output_dir)/'slice_memberships.png' if output_dir else None)
    fig2,axes=plt.subplots(1,2,figsize=(12,3.8));axes[0].bar(np.arange(len(mass)),mass)
    axes[0].axhline(len(A)/A.shape[1],ls='--',color='black',label='Uniform mass');axes[0].legend()
    axes[0].set(xlabel='Slice index',ylabel=mass_label,title='Routing mass')
    maximum=A.max(1);axes[1].hist(maximum,bins=50);axes[1].axvline(1/A.shape[1],ls='--',color='black')
    axes[1].set(xlabel=maximum_label,ylabel='Points',title=concentration_title)
    fig2.tight_layout();_finish(fig2,Path(output_dir)/'slice_distribution.png' if output_dir else None)
    print(f"Routing shape N,H,M={routing['heads'].shape}; learned clamped temperature={routing['temperature']}; row sum={A[0].sum():.6f}")
    return fig,fig2


def plot_routing_entropy(routings, *, title="Per-head routing entropy", output_path=None):
    """Compare correct per-head entropy with entropy of the head mixture."""
    rows=[]
    for routing in routings:
        A=routing['heads'];Am=routing['mean'];M=A.shape[2]
        H=-np.sum(A*np.log2(A+1e-12),axis=2);Hmix=-np.sum(Am*np.log2(Am+1e-12),axis=1)
        rows.append({'block':routing['block_index'],'entropy':H.mean(),'mixture':Hmix.mean(),
                     'max':np.log2(M),'effective':2**H.mean(),'concentration':A.max(2).mean()*M,'point_entropy':H.mean(1)})
    final=rows[-1];coords=routings[-1]['coordinates'];fig=plt.figure(figsize=(14,8))
    ax=fig.add_subplot(2,2,1,projection='3d');art=ax.scatter(*coords.T,c=final['point_entropy'],s=2,cmap='plasma')
    fig.colorbar(art,ax=ax,shrink=.5,label='Per-head entropy (bits)');ax.set_title(f"Sampled-input entropy: block {final['block']}");ax.view_init(20,-55)
    ax=fig.add_subplot(2,2,2);x=np.arange(len(rows))
    ax.plot(x,[r['entropy'] for r in rows],'o-',label='Mean per-head entropy');ax.plot(x,[r['mixture'] for r in rows],'s--',label='Entropy after head averaging')
    ax.axhline(final['max'],color='black',ls=':');ax.set(xticks=x,xticklabels=[str(r['block']) for r in rows],xlabel='Measured block',ylabel='Bits',title='Depth and head-averaging effect');ax.legend(fontsize=8)
    ax=fig.add_subplot(2,2,3);ax.hist(final['point_entropy'],bins=50);ax.set(xlabel='Mean per-head entropy',ylabel='Points',title='Distribution on sampled model inputs')
    ax=fig.add_subplot(2,2,4);z=coords[:,2];edges=np.linspace(z.min(),z.max(),21);indices=np.clip(np.digitize(z,edges)-1,0,19)
    means=[np.mean(final['point_entropy'][indices==i]) if np.any(indices==i) else np.nan for i in range(20)]
    ax.plot((edges[:-1]+edges[1:])/2,means,'o-');ax.set(xlabel='Scaled height z',ylabel='Mean per-head entropy',title='Spatial routing diagnostic')
    fig.suptitle(title);fig.tight_layout();_finish(fig,output_path)
    print(tabulate([[r['block'],r['entropy'],r['mixture'],r['mixture']-r['entropy'],r['effective'],r['concentration']] for r in rows],headers=['Block','Per-head H','Head-mixture H','Jensen gap','Effective slices','Top-1 / uniform'],tablefmt='github',floatfmt='.4f'))
    print('Routing entropy describes membership concentration; assess prediction quality with held-out field and force errors.')
    return rows




























def plot_sample_field_parity(reference,prediction,*,title='Sampled field diagnostic',output_path=None):
    """Pointwise parity when a small training batch has no original mesh faces."""
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,truth,estimate,label in [(axes[0],reference[:,0],prediction[:,0],'Pressure'),
                                   (axes[1],np.linalg.norm(reference[:,1:4],axis=1),np.linalg.norm(prediction[:,1:4],axis=1),'Wall-shear magnitude')]:
        ax.scatter(truth,estimate,s=4,alpha=.25);lo=min(truth.min(),estimate.min());hi=max(truth.max(),estimate.max())
        ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='CFD target',ylabel='Prediction',title=label)
    fig.suptitle(title+'; native units; sampled points, no reconstructed faces');fig.tight_layout()
    return _finish(fig,output_path)








class VelocityOnlyDataPipe:
    """Select raw speed from the external datapipe's exact ``[rho,U]`` pair.

    Geometry centering/scaling and target standardization remain the wrapped
    datapipe's responsibility. This adapter neither changes them nor reads a
    fallback velocity. Saved two-feature checkpoints must not use this adapter.
    """
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __len__(self):
        return len(self.datapipe)

    def __getitem__(self, index):
        batch = dict(self.datapipe[index])
        features = batch['fx']
        if features.ndim != 3 or features.shape[-1] != 2:
            raise ValueError('Velocity-only adapter requires fx (1,N,2) in external [air_density,stream_velocity] order')
        velocity = features[..., 1:2]
        if hasattr(velocity, 'detach'):
            valid = bool(velocity.isfinite().all().item()) and bool((velocity > 0).all().item())
        else:
            valid = bool(np.isfinite(velocity).all()) and bool(np.all(velocity > 0))
        if not valid:
            raise ValueError('Each case requires finite positive raw inlet velocity')
        batch['fx'] = velocity
        return batch

    def unscale_model_targets(self, fields):
        return self.datapipe.unscale_model_targets(fields)

    def __getattr__(self, name):
        return getattr(self.datapipe, name)




def load_velocity_notebook_support():
    """Find this restart's adapter without starting training or using an old run."""
    import importlib.util
    configured=os.environ.get('VELOCITY_NOTEBOOK_SUPPORT')
    if configured and not Path(configured).expanduser().is_file():
        raise FileNotFoundError('The configured pre-spike restart adapter is missing')
    candidates=[Path(configured)] if configured else []
    if not configured and os.environ.get('FULL_DATA_UQ_ROOT'):
        candidates.append(Path(os.environ['FULL_DATA_UQ_ROOT'])/'notebook_support.py')
    seen=set()
    for path in candidates:
        path=path.expanduser().resolve()
        if path in seen or not path.is_file():continue
        seen.add(path)
        spec=importlib.util.spec_from_file_location('_prespike_notebook_support',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if (getattr(module,'PROTOCOL',None)!='full_data_velocity_only_shared_stl_bbox_v1'
                or getattr(module,'EXPERIMENT_ID',None)!='geo_pres173_warmup_cosine_v1'):
            if configured:
                raise ValueError('Set VELOCITY_NOTEBOOK_SUPPORT to this pre-spike restart adapter')
            continue
        return module
    return None


# ===========================================================================
# Field restoration and surface traction integration
# ===========================================================================

"""Small, explicit inference and force helpers used by the surface tutorials.

The trained checkpoints predict four *native* channels: pressure, then the x/y/z
components of wall shear. Training standardization is inverted once; velocity
normalization belongs to the subsequent coefficient calculation, not to field
reconstruction. These helpers deliberately require the pressure convention.
"""



def restore_native_fields(standardized, train_mean, train_std):
    """Invert training standardization, ``native = standardized * std + mean``.

    No density or velocity factor is applied here. The supplied checkpoints were
    trained on native fields, rather than pressure/shear coefficients.
    """
    values = np.asarray(standardized)
    mean, std = np.asarray(train_mean), np.asarray(train_std)
    if mean.shape != (4,) or std.shape != (4,) or values.shape[-1] != 4:
        raise ValueError("Expected four native channels and four training statistics")
    if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Fields and training statistics must be finite; std must be positive")
    return values * std + mean


def resolve_case_flow(case_path, metadata):
    """Read the same per-case Zarr attributes supplied to the trained model.

    ``metadata`` is the corresponding parsed ``case*_info.txt`` mapping, whose
    ``Velocity`` must agree with ``stream_velocity``. This cross-check catches
    accidental use of an older curation with a constant default velocity.
    Zarr v2 and v3 attribute files can be read without importing Zarr.
    """
    path = Path(case_path)
    if (path / ".zattrs").is_file():
        attrs = json.loads((path / ".zattrs").read_text())
    elif (path / "zarr.json").is_file():
        attrs = json.loads((path / "zarr.json").read_text())["attributes"]
    else:
        raise FileNotFoundError("No Zarr attribute file in " + str(path))
    velocity, density = float(attrs["stream_velocity"]), float(attrs["air_density"])
    metadata_velocity = float(metadata["Velocity"])
    if not np.isfinite([velocity, density, metadata_velocity]).all() or min(velocity, density) <= 0:
        raise ValueError("A finite positive case velocity and density are required")
    if not np.isclose(velocity, metadata_velocity, rtol=1e-7, atol=1e-8):
        raise ValueError("Case inlet velocity disagrees with its metadata: {} versus {}".format(velocity, metadata_velocity))
    return {"stream_velocity": velocity, "air_density": density}


def integrate_surface_traction(fields, normals, areas, *, normalise_normals=False):
    """Integrate the upstream PhysicsNeMo sign convention on all three axes.

    Returns ``total``, ``pressure`` and ``shear`` vectors using
    ``sum(area * (p * normal - wall_shear))``. Inputs must have matching point
    association (with tributary point areas) or matching cell association.
    Never mix cell areas with point fields. Raw normals are used by default,
    exactly as in ``compute_force_coefficients`` in the upstream inference
    scripts. Their lengths are checked before use.
    """
    fields, normals, areas = (np.asarray(value, dtype=np.float64) for value in (fields, normals, areas))
    areas = areas.reshape(-1)
    if fields.ndim != 2 or fields.shape[1] != 4 or normals.shape != (len(fields), 3) or areas.shape != (len(fields),):
        raise ValueError("Expected fields (N,4), normals (N,3) and areas (N,)")
    if not all(np.isfinite(value).all() for value in (fields, normals, areas)) or np.any(areas < 0):
        raise ValueError("Finite surface fields, normals and nonnegative areas are required")
    lengths = np.linalg.norm(normals, axis=1)
    active = areas > 0
    if np.any(lengths[active] <= 0) or not np.allclose(lengths[active], 1.0, rtol=1e-4, atol=1e-6):
        raise ValueError("Expected nonzero unit surface normals")
    if normalise_normals:
        normals = normals / np.where(active, lengths, 1.0)[:, None]
    pressure = np.sum(areas[:, None] * fields[:, :1] * normals, axis=0, dtype=np.float64)
    shear = -np.sum(areas[:, None] * fields[:, 1:4], axis=0, dtype=np.float64)
    return {"total": pressure + shear, "pressure": pressure, "shear": shear}


def force_coefficient_components(components, flow, reference_area_m2, *, pressure_convention, symmetry_factor=1.0):
    """Convert native traction integrals to explicitly assumed coefficients.

    ``pressure_convention='kinematic'`` means both pressure and wall shear are
    per unit density; their common density cancels, leaving ``0.5*U**2*A``.
    ``'physical'`` assumes pressure/shear in Pa and uses ``0.5*rho*U**2*A``.
    The current dataset does not declare these units, so either conversion must
    be labelled conditional. For the represented half body, ``symmetry_factor=2``
    assumes x/z traction symmetry and reconstructs the full-body x/z integrals;
    this assumption must not be applied to spanwise force without reflection.
    """
    velocity, density = float(flow["stream_velocity"]), float(flow["air_density"])
    area, multiplier = float(reference_area_m2), float(symmetry_factor)
    if pressure_convention not in ("kinematic", "physical"):
        raise ValueError("Choose pressure_convention='kinematic' or 'physical' explicitly")
    if not np.isfinite([velocity, density, area, multiplier]).all() or min(velocity, density, area, multiplier) <= 0:
        raise ValueError("Positive finite flow, reference area and symmetry factor are required")
    denominator = 0.5 * velocity ** 2 * area
    if pressure_convention == "physical":
        denominator *= density
    return {key: np.asarray(components[key], dtype=np.float64) * multiplier / denominator
            for key in ("total", "pressure", "shear")}


def regression_metrics(target, prediction):
    """Case-wise R² and errors, with target spread to interpret small lift SST."""
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or target.shape != prediction.shape or len(target) < 2:
        raise ValueError("Expected at least two matching scalar case targets and predictions")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression inputs must be finite")
    error = prediction - target
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((target - target.mean()) ** 2))
    return dict(n_cases=len(target), r2=float(1 - sse / sst) if sst > 0 else None,
                rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                bias=float(error.mean()), sse=sse, sst=sst, target_mean=float(target.mean()),
                target_std=float(target.std()), target_min=float(target.min()), target_max=float(target.max()))


# ===========================================================================
# Geometry metadata consistency audit
# ===========================================================================

"""Geometry metadata consistency plots for the curation notebook.

The saved comparison is diagnostic. It never rewrites source metadata or assigns
new experimental roles.
"""


def _audit_directory(directory=None):
    if directory is not None:
        return Path(directory).expanduser().resolve()
    configured = os.environ.get('UQ_CURATION_REPORT_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for start in candidates:
        for ancestor in (start, *start.parents):
            for relative in ('metadata_curation', 'reports/metadata_curation',
                             'uq_only_release/reports/metadata_curation'):
                candidate = ancestor / relative
                if (candidate / 'summary.json').is_file():
                    return candidate
    raise FileNotFoundError('Set UQ_CURATION_REPORT_DIR to the packaged geometry consistency report.')


def plot_metadata_curation_audit(directory=None, output_path=None):
    """Compare all supplied angles with measured meshes, without role labels."""
    import numpy as np
    import matplotlib.pyplot as plt
    path = _audit_directory(directory)
    summary = json.loads((path / 'summary.json').read_text())
    with (path / 'case_geometry.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(summary['n_source_cases']):
        raise ValueError('Geometry comparison rows do not match the recorded case count.')
    supplied = np.array([float(row['metadata_slant_deg']) for row in rows])
    measured = np.array([float(row['slant_deg']) for row in rows])
    if not (np.isfinite(supplied).all() and np.isfinite(measured).all()):
        raise ValueError('Nonfinite geometry angles in the saved audit.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    pair, counts = np.unique(np.c_[supplied, measured], axis=0, return_counts=True)
    points = axes[0].scatter(pair[:, 0], pair[:, 1], s=16 + 12 * counts,
                             c=counts, cmap='Blues', edgecolors='#28547b', linewidths=.4)
    lo = min(supplied.min(), measured.min()) - 2
    hi = max(supplied.max(), measured.max()) + 2
    axes[0].plot([lo, hi], [lo, hi], '--', color='black', linewidth=1,
                 label='Angle agreement')
    axes[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Supplied metadata slant [degrees]',
                ylabel='Measured mesh slant [degrees]', title='All 508 source cases')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.15)
    fig.colorbar(points, ax=axes[0], label='Cases sharing this angle pair', pad=.02)
    values = [summary['n_metadata_slant_disagreements'], summary['n_five_geometry_disagreements']]
    bars = axes[1].bar(['Slant angle', 'Any of five descriptors'], values,
                       color=['#2878a5', '#dc7b36'], width=.6)
    axes[1].bar_label(bars, labels=[f'{value} / 508' for value in values], padding=4)
    axes[1].set(ylim=(0, 260), ylabel='Cases with metadata/mesh disagreement',
                title='Source consistency counts')
    axes[1].grid(axis='y', alpha=.15)
    fig.suptitle('Source metadata audit: original info files remain unchanged', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    print(f"Audit: {len(rows)} cases; {values[0]} slant discrepancies; "
          f"{values[1]} cases differ in at least one geometry descriptor.")
    print('Five descriptors: length, width, height, ground clearance and rear slant.')
    print('Measured rear-plane convention:', summary['measurement_definition'])
    print('Original info files are unchanged; model geometry comes from actual mesh coordinates.')
    plt.show()
    return fig


# ===========================================================================
# GeoTransolver training and validation teaching helpers
# ===========================================================================

"""Small display and fixed-validation utilities for Notebook 5.

Numerical preprocessing delegates to the unchanged native trainer. Production
weights, normalization and calibration files are read-only. Actual teaching
updates are handled separately by the signed notebook-support adapter.
"""


def load_normalization(path, device):
    """Read the four saved training-only target scales without refitting them."""
    import torch
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key:np.array(archive[key], copy=True) for key in ("mean", "std")}
        convention = str(archive["field_convention"]) if "field_convention" in archive else "native field units unverified"
    if any(value.shape != (4,) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Expected finite mean/std arrays for pressure and three shear channels")
    if np.any(arrays["std"] <= 0):
        raise ValueError("Every target standard deviation must be positive")
    factors = {key:torch.as_tensor(value, dtype=torch.float32, device=device) for key,value in arrays.items()}
    print("Device:", device, "| field convention:", convention)
    for name, mean, std in zip(("pressure", "shear-x", "shear-y", "shear-z"), arrays["mean"], arrays["std"]):
        print(f"{name:10s} mean={mean:11.5g}  std={std:11.5g}")
    return arrays, factors


class FixedCaseDataPipe:
    """Reproduce native per-case validation sampling and target normalization.

    A private case-seeded generator fixes both surface and geometry draws. This
    read-only diagnostic adapter never represents dynamic optimization sampling.
    """
    def __init__(self, rows, *, support, settings, device, factors, reference_scale):
        import torch
        self.rows, self.support = rows, support
        self.trainer = support._trainer("geotransolver")
        self.settings, self.factors = settings, factors
        self.context = SimpleNamespace(torch=torch, np=np, device=device, S=settings,
            scale=torch.as_tensor(reference_scale, dtype=torch.float32, device=device),
            mean=factors["mean"], std=factors["std"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        data = self.trainer.read_case(row, np)
        batch = self.support._native_batch(self.trainer, self.context, data,
            seed=int(self.settings["seed"])+int(row["case_id"]))
        return dict(fx=batch["scalar"], embeddings=batch["local"],
            geometry=batch["geometry"], fields=batch["target"])

    def unscale_model_targets(self, fields):
        return fields*self.factors["std"]+self.factors["mean"]


def field_metrics(prediction, target):
    """Native-field errors; these are neither force scores nor uncertainty estimates."""
    import torch
    relative_l2 = (((prediction-target).square().sum(dim=1)).sqrt()
                   / target.square().sum(dim=1).sqrt().clamp_min(1e-8)).mean(dim=0)
    shear_pred, shear_target = prediction[..., 1:].norm(dim=-1), target[..., 1:].norm(dim=-1)
    result = {name:float(value) for name,value in zip(
        ("relative_l2_pressure", "relative_l2_shear_x", "relative_l2_shear_y", "relative_l2_shear_z"), relative_l2)}
    result["relative_l1_shear_magnitude"] = float((shear_pred-shear_target).abs().sum()
        / shear_target.abs().sum().clamp_min(1e-8))
    return result


def plot_geo_input_flow(reference_scale, *, surface_points, geometry_points, output_path):
    """One input-contract diagram; interval calibration is taught in Notebook 6."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.5, 3.4)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    boxes = [(.02,.68,.32,.23,f"Surface positions + unit normals\nShared frame → (1, {surface_points}, 6)","#e7f0f8"),
             (.02,.37,.32,.23,f"STL context vertices\nSame frame → (1, {geometry_points}, 3)","#e8f3ed"),
             (.02,.06,.32,.23,"This case's recorded inlet speed\nU_case → (1, 1, 1)","#fff0df"),
             (.48,.35,.20,.31,"GeoTransolver\nGALE blocks\nSaved constructor", "#eef0f4"),
             (.79,.35,.19,.31,f"Pressure + shear xyz\n(1, {surface_points}, 4)\nTraining z-scores", "#f2eaf4")]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",facecolor=color,edgecolor="#607080"))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    for y in (.795,.485,.175):
        ax.annotate("",xy=(.47,.505),xytext=(.35,y),arrowprops=dict(arrowstyle="->",color="#526477"))
    ax.annotate("",xy=(.78,.505),xytext=(.69,.505),arrowprops=dict(arrowstyle="->",color="#526477"))
    scale = ", ".join(f"{value:.5g}" for value in reference_scale)
    ax.set_title("One case, three inputs: velocity is separate from geometry",fontsize=12)
    fig.text(.5,.01,f"Both coordinate inputs use (x − case mean STL triangle center) / s; s = [{scale}] m from all 508 STLs.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1)); path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,bbox_inches="tight"); plt.show(); return fig


# ===========================================================================
# Calibrated pointwise and global drag uncertainty
# ===========================================================================

"""Small display and plotting helpers for the calibrated-UQ tutorial release.

Numerical reports are recomputed from the completed experiment by the separate
report builder. These helpers use the shared 30-case validation/evaluation partition throughout.
"""


_MODELS = {"transolver": "Transolver", "geotransolver": "GeoTransolver"}
_EXPERIMENT_ID = "geo_pres173_warmup_cosine_v1"
_TARGET_EPOCHS = {"transolver":600,"geotransolver":500}


def _reports():
    configured = os.environ.get("UQ_ONLY_RESULTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    adjacent = Path(__file__).resolve().parent / "uq_results"
    if adjacent.is_dir():
        return adjacent
    external = os.environ.get('FULL_DATA_UQ_ROOT')
    if external:
        return Path(external).expanduser().resolve()/'reports'
    raise FileNotFoundError('Set UQ_ONLY_RESULTS_DIR to the matching external experiment reports for a live data run.')


def _figure_cache():
    """Keep new renders in staging; never write into an earlier report/release."""
    configured = os.environ.get("UQ_PREVIEW_FIGURES_DIR")
    path = Path(configured).expanduser().resolve() if configured else Path(os.environ.get("NOTEBOOK_OUTPUT_DIR", "."))/"fig"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _result_note(result):
    return "Final checkpoints: Transolver 600 / GeoTransolver 500 epochs; budgets differ; 30 development cases."


def show_uq_result_source():
    from IPython.display import Markdown, display
    result = load_uq_results()
    display(Markdown("**Result source:** "+_result_note(result)))
    return result


def _case_rows(result):
    path = _reports()/"case_metrics.csv"
    proof = json.loads((_reports()/"verification.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != proof["files"][path.name]:
        raise ValueError("The case table changed after report verification.")
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    expected = set(map(str, result["validation_evaluation_case_ids"]))
    for model in _MODELS:
        selected = [row for row in records if row["model"] == model]
        if (len(selected) != 30 or {row["case_id"] for row in selected} != expected
                or any(row["role"] != "validation_evaluation" for row in selected)):
            raise ValueError("Expected exactly the same 30 development cases per model.")
    return records


def _load_case(model, case_id, result):
    """Validate archive/checkpoint hashes before constructing additional views."""
    case_id = str(case_id)
    if model not in _MODELS or case_id not in result["validation_evaluation_case_ids"]:
        raise ValueError("Choose a declared model and validation/evaluation case.")
    root = _evaluation_root(model)
    proof = json.loads((_reports()/"verification.json").read_text())["models"][model]
    manifest_path = root/"manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != proof["source_manifest_sha256"]:
        raise ValueError("The evaluation manifest differs from the audited report.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["checkpoint_sha256"] != result["models"][model]["checkpoint_sha256"]:
        raise ValueError("Prediction archives and displayed results use different checkpoints.")
    entry = next(row for row in manifest["cases"] if str(row["case_id"]) == case_id)
    path = (root/entry["path"]).resolve()
    if root.resolve() not in path.parents or entry["split"] != "validation":
        raise ValueError("Archive must be inside the declared evaluation directory.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != proof["verified_archives"][case_id]["archive_sha256"]:
        raise ValueError("The prediction archive changed after verification.")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def _notebook_support():
    """The new adapter routes inherited Transolver and continued Geo separately."""
    support = load_velocity_notebook_support()
    if support is None:
        raise RuntimeError("The matching notebook support adapter is required.")
    return support


def _evaluation_root(model):
    direct = _production()/"evaluation"/model
    if os.environ.get("FULL_DATA_UQ_ROOT") and direct.is_dir():
        return direct
    try:
        support = _notebook_support()
    except (ImportError, RuntimeError):
        if os.environ.get("VELOCITY_NOTEBOOK_SUPPORT"):
            raise
        support = None
    if support is not None and hasattr(support, "evaluation_directory"):
        return Path(support.evaluation_directory(model))
    return direct


def _production():
    configured = os.environ.get("FULL_DATA_UQ_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    report_root = _reports().parent
    if (report_root/"evaluation").is_dir():
        return report_root
    raise FileNotFoundError("Set FULL_DATA_UQ_ROOT to the completed experiment directory.")


def load_uq_results():
    """Read the completed, shared-development report and reject incompatible models."""
    path = _reports() / "summary.json"
    ready = json.loads((_reports()/"READY.json").read_text())
    verification_path = _reports()/"verification.json"
    if (ready.get("status") != "complete"
            or hashlib.sha256(verification_path.read_bytes()).hexdigest() != ready["verification_sha256"]):
        raise ValueError("The completed report verification record changed.")
    verification = json.loads(verification_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != verification["files"]["summary.json"]:
        raise ValueError("The report summary differs from its verified version.")
    result = json.loads(path.read_text())
    if (result.get("status") != "complete"
            or result.get("preprocessing_protocol") != "full_data_velocity_only_shared_stl_bbox_v1"
            or result.get("selection_and_evaluation_reused") is not True
            or result.get("independent_test_available") is not False
            or [result.get(key) for key in ("train_cases", "validation_evaluation_cases", "calibration_cases")] != [458, 30, 20]):
        raise ValueError("A completed calibrated-UQ report is required.")
    if result.get('evaluated_epochs') != _TARGET_EPOCHS:
        raise ValueError('The report must identify each model\'s separate final epoch budget')
    for name in ("transolver", "geotransolver"):
        record = result["models"][name]
        if int(record["selected_epoch"]) != _TARGET_EPOCHS[name] or int(record["trained_epochs"]) != _TARGET_EPOCHS[name]:
            raise ValueError("The teaching report and completed model provenance differ.")
    if result.get('checkpoint_selection_policy') != 'last_completed_epoch':
        raise ValueError('Production reports require the last-completed-epoch checkpoint policy')
    for model,record in result['models'].items():
        if (record.get('checkpoint_selection_policy') != 'last_completed_epoch'
                or record.get('evaluated_epoch') != _TARGET_EPOCHS[model] or record.get('selected_epoch') != _TARGET_EPOCHS[model]):
            raise ValueError('Each production model must evaluate its declared final completed epoch')
    geo=result['models']['geotransolver']
    if (geo.get('training_recipe_id') != _EXPERIMENT_ID
            or geo.get('inherited_epochs') != 6 or geo.get('phase_training_epochs') != 494
            or geo.get('continuation_provenance',{}).get('parent_epoch') != 6):
        raise ValueError('This notebook requires fresh final-500 Geo results from the verified 6+494 restart')
    return result


def _display_png(filename):
    from IPython.display import Image, display
    path = _reports() / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    verified = json.loads((_reports()/"verification.json").read_text())["files"]
    if filename in verified and hashlib.sha256(path.read_bytes()).hexdigest() != verified[filename]:
        raise ValueError("A verified scientific figure changed: " + filename)
    display(Image(filename=str(path)))


def _selected(result, model):
    if model is not None and model not in _MODELS:
        raise ValueError("model must be transolver, geotransolver, or None")
    return [(key, result["models"][key]) for key in _MODELS if model in (None, key)]


def show_uq_drag(model=None):
    """Show native drag accuracy on the same 30 shared validation/evaluation cases."""
    from IPython.display import Markdown, display
    result = load_uq_results()
    lines = ["| Model | Evaluated epoch | Drag R² | Drag RMSE | Drag MAE |",
             "|---|---:|---:|---:|---:|"]
    for key, record in _selected(result, model):
        metric = record["drag"]
        lines.append(f"| {_MODELS[key]} | {record['selected_epoch']} | {metric['r2']:.4f} | "
                     f"{metric['rmse']:.4f} | {metric['mae']:.4f} |")
    display(Markdown("**30 shared validation/evaluation cases**; native signed drag integrals.\n\n" + "\n".join(lines)))
    _display_png("drag_parity.png")


def show_uq_summary(model=None):
    """Print coverage and precision together, with no fabricated deterministic spread."""
    from IPython.display import Markdown, display
    result = load_uq_results()
    lines = ["| Model | Quantity | Interval | Coverage | Mean full width | 90% interval score ↓ |",
             "|---|---|---|---:|---:|---:|"]
    for key, record in _selected(result, model):
        for quantity in ("pressure", "drag"):
            for method in ("raw", "calibrated"):
                metric = record["uq"][quantity].get(method)
                if metric is None:
                    continue
                label = "Raw MC ±1.645s" if method == "raw" else "Calibrated"
                quantity_label = "Pressure (vertices)" if quantity == "pressure" else "Drag (cases)"
                lines.append(f"| {_MODELS[key]} | {quantity_label} | {label} | {metric['coverage']:.1%} | "
                             f"{metric['full_width']:.4g} | {metric['interval_score_90']:.4g} |")
    display(Markdown("**30 shared validation/evaluation cases**; calibration uses **20 separate cases**. "
                     "Pressure summaries give each case equal weight. Width and score retain each quantity's native units.\n\n"
                     + "\n".join(lines)))


def show_uq_figures():
    """Display the independently generated pressure and drag calibration reports."""
    for filename in ("pressure_uq.png", "drag_uq.png", "calibrated_reliability.png"):
        _display_png(filename)


def show_uq_figure(name):
    if name not in {"pressure_uq", "drag_uq", "calibrated_reliability", "training_curves"}:
        raise ValueError("Choose one of the completed tutorial report figures.")
    _display_png(name+".png")


def plot_uq_workflow():
    """Draw the four role-specific decisions without exposing plotting scaffolding."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set(xlim=(0, 4), ylim=(0, 1)); ax.axis("off")
    titles = ["458 training cases", "30 validation cases", "20 calibration cases", "Same 30 validation cases"]
    purposes = ["Fit weights + field scaling", "Freeze Trans 600 / Geo 500", "Fit interval scales", "Report development results"]
    for i, (title, purpose) in enumerate(zip(titles, purposes)):
        ax.add_patch(FancyBboxPatch((i + .04, .23), .90, .55,
                     boxstyle="round,pad=0.02", facecolor="#e8f1fa", edgecolor="#4477aa"))
        ax.text(i+.49, .61, title, ha="center", fontsize=10, fontweight="bold")
        ax.text(i+.49, .40, purpose, ha="center", fontsize=9)
        if i < 3:
            ax.add_patch(FancyArrowPatch((i+.96, .5), (i+1.035, .5), arrowstyle="->", mutation_scale=12))
    ax.text(2, .07, "Training/calibration/validation IDs are disjoint. Evaluation reuses validation; no independent test set.", ha="center", fontsize=10)
    fig.tight_layout(); plt.show()


def plot_toy_predictions(x_train, y_train, x_test, mean, std, multiplier=1.645):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.scatter(x_train, y_train, s=20, c="#444444", alpha=.6, label="Training observations")
    ax.plot(x_test, np.sin(x_test), "k--", lw=1.2, label="Known toy mean")
    ax.plot(x_test, mean, color="#4477aa", label="MC predictive mean")
    ax.fill_between(x_test, mean-multiplier*std, mean+multiplier*std, color="#4477aa", alpha=.22,
                    label=f"Mean ± {multiplier:.3g} × MC standard deviation")
    ax.set(xlabel="Toy input x", ylabel="Toy response", title="A fitted stochastic model: mean and raw spread")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); plt.show()


def plot_toy_calibration(target, mean, std, multiplier):
    import matplotlib.pyplot as plt
    error = np.abs(np.asarray(target)-mean)
    halfwidths = [1.645*std, multiplier*std]
    coverage = [np.mean(error <= width) for width in halfwidths]
    widths = [np.mean(2*width) for width in halfwidths]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    labels = ["Raw MC spread", "Calibration-scaled"]
    axes[0].bar(labels, np.array(coverage)*100, color=["#bbbbbb", "#4477aa"])
    axes[0].axhline(90, color="#cc6677", linestyle="--", label="Nominal 90%")
    axes[0].set(ylabel="Toy evaluation observations covered [%]", ylim=(0, 105)); axes[0].legend(fontsize=8)
    axes[1].bar(labels, widths, color=["#bbbbbb", "#4477aa"])
    axes[1].set(ylabel="Mean full interval width [toy units]")
    fig.suptitle("Toy calibration: inspect containment and precision together")
    fig.tight_layout(); plt.show()


def show_training_evidence(model="geotransolver", include_demo=True):
    """Read each model's actual history and recorded full-epoch continuation."""
    result = load_uq_results()
    record = result["models"][model]
    target = _TARGET_EPOCHS[model]
    exported = _figure_cache() / f"{model}_training_evidence.json"
    evidence = json.loads(exported.read_text()) if exported.is_file() else {}
    if evidence.get("checkpoint_sha256") != record["checkpoint_sha256"]:
        root = _production()
        support = _notebook_support()
        evaluation_root = Path(support.evaluation_directory(model)) if hasattr(support, "evaluation_directory") else root/"evaluation"/model
        demo_root = Path(support.teaching_demo_directory(model))
        history = json.loads((evaluation_root / "learning_curve.json").read_text())
        demo = json.loads((demo_root / "notebook_demo_summary.json").read_text())
        evidence = {"learning_curve": history, "demo": demo, "checkpoint_sha256": record["checkpoint_sha256"]}
        exported.write_text(json.dumps(evidence, indent=2)+"\n")
    history, demo = evidence["learning_curve"], evidence["demo"]
    if len(history) != target or [r["epoch"] for r in history] != list(range(1, target+1)):
        raise ValueError("Training evidence must contain every actual epoch for this model's budget.")
    if (demo.get("status") != "complete" or not demo.get("source_artifacts_unchanged")
            or demo["source_checkpoint_epoch"] != target or demo["final_demo_epoch"] != target+4):
        raise ValueError("The isolated teaching continuation did not preserve the production run.")
    rows = demo["epochs"]
    if [r["epoch"] for r in rows] != list(range(target+1,target+5)):
        raise ValueError("Teaching output must contain four actual epochs after this model's final state.")
    if any(r["train_cases"] != 458 or r["validation_cases"] != 30 for r in rows):
        raise ValueError("The continuation must visit the full training and fixed-validation partitions.")
    plot_measured_training(history, selected_epoch=record["selected_epoch"],
                           continuation=rows if include_demo else None,
                           title=f"{_MODELS[model]}: {target} real training epochs")
    print(f"Prediction weights: evaluated epoch {record['selected_epoch']}; trained through {target}.")
    if include_demo:
        print(f"RECORDED OUTPUT of actual isolated optimizer epochs {target+1}–{target+4}; no new updates on this notebook rerun.")
        print("Epoch  Training MSE  Validation MSE  LR used       Seconds")
        for row in rows:
            lr = row.get("learning_rate_used", row.get("learning_rates"))[0]
            print(f"{row['epoch']:5d}  {row['train_mse']:.6f}      {row['validation_mse']:.6f}       {lr:.4g}  {row['seconds']:.1f}")
        print("Production source artifacts stayed unchanged; these teaching weights supply no reported evaluation results.")


def run_actual_training_demo(model="geotransolver"):
    """Perform or safely replay four real optimizer epochs after the final state."""
    support = _notebook_support()
    if support is None or support.PROTOCOL != "full_data_velocity_only_shared_stl_bbox_v1":
        raise RuntimeError("The matching full-data notebook adapter is required.")
    bundle = support.discover_completed_experiment(model)
    if bundle is None or bundle.get("pilot"):
        raise ValueError("The full completed experiment is required for actual teaching epochs.")
    demo_root = Path(support.teaching_demo_directory(model))
    support.run_demo(bundle["training_dir"], demo_root,
                     additional_epochs=4, allow_pilot=False, return_model=False)
    show_training_evidence(model, include_demo=True)


def show_uq_surface_examples(model="geotransolver", case_ids=None, *, shape=None, show=True):
    """Display audited original-mesh pressure/UQ maps without Torch or VTK.

    The first default case (921) is a selected low-error illustration. Cases 51
    and 604 retain the earlier examples so the teaching notebook shows more
    than one level of accuracy. Aggregate metrics still use all 30 cases.
    A ``(2,2)`` layout is useful for a post; the default is a wide four-panel row.
    """
    if show:
        from IPython.display import Image, display
    if model not in _MODELS:
        raise ValueError("Unknown surface model")
    result = load_uq_results()
    case_ids = tuple(map(str, case_ids if case_ids is not None else ("921", "51", "604")))
    if not set(case_ids) <= set(map(str, result["validation_evaluation_case_ids"])):
        raise ValueError("Surface examples must use declared shared validation/evaluation cases")
    shape = tuple(shape or (1, 4))
    proof = json.loads((_reports()/"verification.json").read_text())["models"][model]
    q = float(result["models"][model]["calibration"]["pressure_q"])
    audits = []
    for case_id in case_ids:
        suffix = "" if shape == (1, 4) else "_"+"x".join(map(str, shape))
        image_path = _figure_cache()/f"surface_{model}_case{int(case_id)}{suffix}.png"
        metadata_path = image_path.with_suffix(".json")
        expected = dict(model=model, case_id=case_id,
            checkpoint_sha256=result["models"][model]["checkpoint_sha256"],
            selected_epoch=result["models"][model]["selected_epoch"],
            archive_sha256=proof["verified_archives"][case_id]["archive_sha256"],
            render_schema=6, pressure_display_percentiles=[5, 95], size_color_quantile=.99,
            colormaps=["turbo", "turbo", "viridis", "magma"], layout=list(shape),
            pressure_calibration_multiplier=q, point_std_floor=1e-8,
            display_association="Mean of original vertex field values on each original polygon")
        cached = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        cache_valid = (image_path.is_file() and all(cached.get(key) == value for key, value in expected.items())
            and cached.get("png_sha256") == hashlib.sha256(image_path.read_bytes()).hexdigest())
        if not cache_valid:
            data = _load_case(model, case_id, result)
            target, mean, std = data["target"][:,0], data["mean"][:,0], data["std"][:,0]
            halfwidth = q*np.maximum(std, 1e-8) if model == "geotransolver" else np.full_like(mean, q)
            fig, audit = render_surface_comparison(data["coordinates"],
                data["cell_connectivity"], data["cell_offsets"], target, mean,
                halfwidth=halfwidth, model_label=_MODELS[model], case_id=case_id,
                epoch=expected["selected_epoch"], speed=float(data["stream_velocity"]),
                shape=shape, output_path=image_path, show=False)
            plt.close(fig)
            cached = dict(audit, **expected)
            cached.update(target_association="Original verified VTP PointData in original vertex order",
                plot_scope="Shared validation/development case; original polygon display; original-cell force reference separate")
            metadata_path.write_text(json.dumps(cached, indent=2)+"\n")
        print(f"{_MODELS[model]} case {case_id}: full-surface pressure RMSE={cached['pressure_rmse']:.5g}; "
              f"relative L2={cached['pressure_relative_l2']:.2%}; "
              f"actual maximum error={cached['full_surface_error_max']:.5g} native units; "
              f"display error/half-width cap={cached['panels'][2]['display_range'][1]:.5g}; "
              f"calibrated point coverage={cached['calibrated_point_coverage']:.3f}")
        panel_audits = cached["panels"]
        print("Display-saturated vertices: "
              f"CFD pressure {panel_audits[0]['fraction_saturated']:.2%}, "
              f"predicted pressure {panel_audits[1]['fraction_saturated']:.2%}, "
              f"error {panel_audits[2]['fraction_above_range']:.2%}, "
              f"half-width {panel_audits[3]['fraction_above_range']:.2%}. "
              "These vertices keep their original values in every metric.")
        if show:
            display(Image(filename=str(image_path)))
        audits.append(cached)
    return audits


def show_uq_pipeline(*, training_cases=458, calibration_cases=20,
                     reporting_cases=30, transolver_epoch=600,
                     geotransolver_epoch=500, mc_samples=20):
    """Draw the study workflow from its stated case counts and model budgets.

    This is a schematic, so it uses these explicit values rather than loading
    experiment reports. The measured accuracy and coverage are explained in
    the notebook's embedded result tables and figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 5.1))
    ax.set(xlim=(-.14, 12), ylim=(0, 5)); ax.axis("off")
    def box(x, y, text, color="#e8f1fa"):
        ax.add_patch(FancyBboxPatch((x, y), 2.55, 1.10, boxstyle="round,pad=.06",
                     facecolor=color, edgecolor="#55718c"))
        ax.text(x+1.275, y+.55, text, ha="center", va="center", fontsize=10)
    def arrow(x, y):
        ax.add_patch(FancyArrowPatch((x, y), (x+.40, y), arrowstyle="->", mutation_scale=15))
    ax.text(0, 4.65, "GeoTransolver: learn the model, measure spread, then calibrate", fontsize=12, weight="bold")
    boxes = [f"{training_cases} training cases\nFit weights and dropout p\nFreeze epoch {geotransolver_epoch}",
             f"{mc_samples} stochastic field passes\nSame case and geometry\nCompute mean μ and spread s",
             f"{calibration_cases} separate calibration cases\nCompare CFD errors with s\nFit pressure q and drag q",
             "Reported intervals\nPressure: μₚ ± qₚsₚ\nDrag: μᴅ ± qᴅsᴅ"]
    for i, text in enumerate(boxes):
        box(i*3., 3.10, text, "#fff0da" if i == 2 else "#e8f1fa")
        if i < 3: arrow(i*3.+2.57, 3.65)
    ax.text(0, 2.68, "Learned p controls stochastic execution. Calibration q uses reference errors and changes interval width.", fontsize=10)
    ax.text(0, 2.15, "Deterministic Transolver: a residual-calibration baseline", fontsize=12, weight="bold")
    baseline = [f"{training_cases} training cases\nFit mean model; freeze epoch {transolver_epoch}",
                "One deterministic field\nNo raw MC spread",
                f"{calibration_cases} separate calibration cases\nQuantile of absolute errors",
                "Reported intervals\nMean ± fixed residual quantile\nSeparate pressure/drag scales"]
    for i, text in enumerate(baseline):
        box(i*3., .60, text, "#fff0da" if i == 2 else "#edf3eb")
        if i < 3: arrow(i*3.+2.57, 1.15)
    fig.suptitle("Two stages: learn stochastic spread, then calibrate interval scale", fontsize=14)
    fig.text(.5, .015, f"Study checkpoints: Transolver {transolver_epoch} / GeoTransolver {geotransolver_epoch} epochs; "
             f"different training budgets; {reporting_cases} development cases.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); plt.show()


def show_uq_drag_intervals():
    """Plot each actual case interval around its error relative to the CFD integral."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    result = load_uq_results()
    rows = _case_rows(result)
    lookup = {(r["model"], r["case_id"]): r for r in rows}
    ids = sorted(result["validation_evaluation_case_ids"],
                 key=lambda cid: float(lookup[("transolver", cid)]["drag_target"]))
    x = np.arange(len(ids))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.0), sharex=True, sharey=True)
    for ax, model in zip(axes, _MODELS):
        selected = [lookup[(model, cid)] for cid in ids]
        error = np.array([float(row["drag_prediction"])-float(row["drag_target"]) for row in selected])
        half = np.array([float(row["drag_calibrated_full_width"])/2 for row in selected])
        covered = np.abs(error) <= half
        ax.axhline(0, color="#333333", lw=1, label="CFD reference")
        ax.errorbar(x, error, yerr=half, fmt="none", ecolor="#7ba3c6", capsize=3, lw=1.5)
        ax.scatter(x, error, s=28, c=np.where(covered, "#4477aa", "#cc3311"), zorder=4)
        if model == "geotransolver":
            raw = np.array([float(row["drag_raw_full_width"])/2 for row in selected])
            ax.errorbar(x, error, yerr=raw, fmt="none", ecolor="#222222", capsize=2, lw=1.2, zorder=5)
        metric = result["models"][model]["uq"]["drag"]["calibrated"]
        ax.set_title(f"{_MODELS[model]} — calibrated: {covered.sum()}/30 contained; "
                     f"mean full width {metric['full_width']:.3f}; score {metric['interval_score_90']:.3f}", loc="left", fontsize=11)
        ax.set_ylabel("Predicted drag − CFD\n[native integral units]")
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(ids, rotation=60, ha="right", fontsize=8)
    axes[-1].set_xlabel("Same 30 cases, ordered by CFD drag; tick labels are case IDs")
    handles = [Line2D([], [], color="#7ba3c6", lw=2, label="Calibrated 90% interval"),
               Line2D([], [], color="#222222", lw=1.2, label="Raw Geo MC band"),
               Line2D([], [], color="#4477aa", marker="o", lw=0, label="Calibrated interval contains reference"),
               Line2D([], [], color="#cc3311", marker="o", lw=0, label="Calibrated interval misses reference")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, fontsize=9)
    fig.suptitle("Whole-case drag uncertainty: an interval succeeds when it crosses the zero-reference line", fontsize=13, y=1.005)
    fig.text(.5, .005, _result_note(result), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .865)); plt.show()


def show_uq_pointwise_diagnostics(model="geotransolver", case_id=None):
    """Compare error with half-width directly, plus the aggregate coverage/width tradeoff."""
    import matplotlib.pyplot as plt
    result = load_uq_results()
    case_id = str(case_id or result["validation_evaluation_case_ids"][0])
    data = _load_case(model, case_id, result)
    target, mean, std = data["target"][:, 0], data["mean"][:, 0], data["std"][:, 0]
    error = np.abs(target-mean)
    q = result["models"][model]["calibration"]["pressure_q"]
    half = q*np.maximum(std, 1e-8) if model == "geotransolver" else np.full_like(mean, q)
    raw = 1.645*std if model == "geotransolver" else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # Complete point arrays enter the hexbin: no error-ranked sampling or clipping.
    density = axes[0].hexbin(half, error, gridsize=45, bins="log", mincnt=1, cmap="viridis")
    upper = float(max(half.max(), error.max()))
    axes[0].plot([0, upper], [0, upper], "--", color="#cc3311", lw=1.2)
    axes[0].set(xlim=(0, upper*1.02), ylim=(0, upper*1.02),
        xlabel="Calibrated pressure half-width [native]", ylabel="Absolute pressure error [native]",
        title=f"Case {case_id}: {np.mean(error <= half):.1%} of vertices contained")
    axes[0].text(.03, .95, "Above diagonal: interval misses CFD\nBelow diagonal: interval contains CFD",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    fig.colorbar(density, ax=axes[0], label="Number of vertices per bin (log colors)")
    for key, method, label, color, marker in (
            ("geotransolver", "raw", "Geo: raw MC", "#999999", "o"),
            ("geotransolver", "calibrated", "Geo: calibrated", "#4477aa", "o"),
            ("transolver", "calibrated", "Transolver: calibrated baseline", "#228833", "s")):
        metric = result["models"][key]["uq"]["pressure"][method]
        axes[1].scatter(metric["full_width"], 100*metric["coverage"], color=color, marker=marker, s=70, label=label)
    axes[1].axhline(90, color="#cc6677", linestyle="--", label="Nominal 90%")
    axes[1].set(xlabel="Mean full pressure interval width [native]", ylabel="Mean per-case vertex coverage [%]",
                ylim=(0, 105), title="All 30 development cases: containment versus width")
    axes[1].legend(loc="lower right", fontsize=8); axes[1].grid(alpha=.2)
    if raw is not None:
        detail = f"Case {case_id}: raw {np.mean(error <= raw):.1%} → calibrated {np.mean(error <= half):.1%}; {len(error):,} vertices."
    else:
        detail = f"Case {case_id}: deterministic residual intervals; {len(error):,} vertices."
    fig.suptitle("Pointwise UQ: compare errors and widths in the same units", fontsize=13)
    fig.text(.5, .045, detail, ha="center", fontsize=9)
    fig.text(.5, .005, _result_note(result), ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .94)); plt.show()


def show_uq_force_workflow(case_id=None):
    """Explain the integration sequence and display the real stored joint-draw integrals."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    result = load_uq_results()
    case_id = str(case_id or result["validation_evaluation_case_ids"][0])
    data = _load_case("geotransolver", case_id, result)
    samples = np.asarray(data["global_samples"][:, 0], dtype=np.float64)
    reference = float(data["global_target"][0])
    mean, std = samples.mean(), samples.std(ddof=1)
    raw = 1.645*std
    q = result["models"]["geotransolver"]["calibration"]["drag_q"]
    calibrated = q*max(std, 1e-12)
    fig = plt.figure(figsize=(12.5, 6.7))
    grid = fig.add_gridspec(2, 2, height_ratios=(.85, 1.3))
    flow = fig.add_subplot(grid[0, :]); flow.set(xlim=(-.14, 12), ylim=(0, 2)); flow.axis("off")
    labels = ["One complete MC draw t\nPressure + shear at vertices",
              "Original mesh cells\nAverage each draw to cells",
              "Cell traction × cell area\nSum signed contributions",
              "One drag value per draw\nRepeat t = 1,…,T"]
    for i, text in enumerate(labels):
        flow.add_patch(FancyBboxPatch((i*3.+.03, .4), 2.52, 1.03, boxstyle="round,pad=.05",
                       facecolor="#e8f1fa", edgecolor="#55718c"))
        flow.text(i*3.+1.29, .915, text, ha="center", va="center", fontsize=9)
        if i < 3:
            flow.add_patch(FancyArrowPatch((i*3.+2.61, .915), (i*3.+2.96, .915),
                          arrowstyle="->", mutation_scale=13))
    flow.text(6, .05, "Schematic procedure. The two panels below use actual archived draws; no synthetic force samples.", ha="center", fontsize=9)
    draws = fig.add_subplot(grid[1, 0])
    draws.scatter(np.arange(1, len(samples)+1), samples, color="#4477aa", s=32)
    draws.axhline(mean, color="#333333", lw=1, label="MC mean")
    draws.set(xlabel="Stochastic field draw index", ylabel="Integrated drag [native units]",
              title=f"Case {case_id}: {len(samples)} measured integrated draws")
    draws.set_xticks(np.unique(np.linspace(1, len(samples), min(5, len(samples)), dtype=int)))
    draws.ticklabel_format(useOffset=False, axis="y")
    draws.legend(fontsize=9); draws.grid(alpha=.2)
    intervals = fig.add_subplot(grid[1, 1])
    residual = mean-reference
    intervals.axvline(0, color="#333333", lw=1, label="CFD reference")
    intervals.errorbar([residual, residual], [1, 0], xerr=[raw, calibrated],
                      fmt="o", color="#4477aa", capsize=5, lw=2)
    intervals.set(yticks=[1, 0], yticklabels=["Raw ±1.645sᴅ", "Calibrated ±qᴅsᴅ"],
                  xlabel="Predicted drag − CFD [native units]", ylim=(-.65, 1.65),
                  title="Same central prediction; different interval scale")
    intervals.grid(axis="x", alpha=.2)
    intervals.text(.02, .98, f"sᴅ = {std:.4g}; qᴅ = {q:.3g}\nCalibrated full width = {2*calibrated:.3g}",
                   transform=intervals.transAxes, va="top", fontsize=9)
    fig.suptitle("Global uncertainty: integrate each joint field draw before measuring its spread", fontsize=13)
    fig.text(.5, .015, _result_note(result), ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95), h_pad=1.3); plt.show()


def show_training_interpretation(model="geotransolver"):
    """Contrast the evaluated final state with historical validation-best diagnostics."""
    from IPython.display import Markdown, display
    result = load_uq_results()
    record = result["models"][model]
    curve = record["learning_curve"]
    evaluated = next(row for row in curve if row["epoch"] == record["selected_epoch"])
    terminal = curve[-1]
    best = min(curve, key=lambda row: row['validation_mse'])
    ratio = terminal["validation_mse"]/best["validation_mse"]
    text = (f"**{_MODELS[model]}:** evaluated epoch **{record['selected_epoch']}**, "
            f"fixed-validation MSE **{evaluated['validation_mse']:.6f}**. "
            f"Historical best validation epoch **{best['epoch']}**, MSE **{best['validation_mse']:.6f}**, "
            f"is a diagnostic only; terminal/best loss ratio **{ratio:.2f}×**. ")
    if ratio > 1.5:
        text += "The terminal loss is substantially worse; this curve does not demonstrate stable convergence. "
    else:
        text += "Use the late curve to judge stability; finishing the epoch budget alone does not establish convergence. "
    display(Markdown(text+"Production accuracy/UQ uses each model's declared final checkpoint (Transolver 600, GeoTransolver 500), even when an earlier epoch has lower validation loss.\n\n*"+_result_note(result)+"*"))


def show_uq_interpretation(quantity):
    """Keep educational numerical captions synchronized with the audited report."""
    from IPython.display import Markdown, display
    if quantity not in ("pressure", "drag"):
        raise ValueError("Choose pressure or drag")
    result = load_uq_results()
    trans = result["models"]["transolver"]["uq"][quantity]["calibrated"]
    geo = result["models"]["geotransolver"]["uq"][quantity]["calibrated"]
    raw = result["models"]["geotransolver"]["uq"][quantity]["raw"]
    if quantity == "pressure":
        text = (f"**Measured pointwise result:** Geo raw bands contain **{raw['coverage']:.1%}** "
                f"of vertices on average per case; calibration changes this to **{geo['coverage']:.1%}**. "
                f"The deterministic Transolver baseline contains **{trans['coverage']:.1%}**. "
                f"Mean full widths are **{geo['full_width']:.3f}** (Geo) and "
                f"**{trans['full_width']:.3f}** (Transolver), in native pressure units. "
                "These are marginal vertex averages, not whole-surface containment.")
    else:
        count = lambda metric: int(round(30*metric["coverage"]))
        text = (f"**Measured whole-case result:** Geo raw bands contain **{count(raw)}/30** references; "
                f"calibration contains **{count(geo)}/30 ({geo['coverage']:.1%})**. "
                f"Transolver's residual baseline contains **{count(trans)}/30 ({trans['coverage']:.1%})**. "
                f"Mean full width: **{geo['full_width']:.3f}** versus **{trans['full_width']:.3f}**; "
                f"proper 90% interval score: **{geo['interval_score_90']:.3f}** versus "
                f"**{trans['interval_score_90']:.3f}** (Geo versus Transolver; lower score is better). "
                "Broad intervals can contain more references without providing the sharper useful answer. "
                "Judge containment, width and score together rather than declaring a winner from coverage alone.")
    display(Markdown(text+"\n\n*"+_result_note(result)+"*"))


def show_uq_coverage_surface(model="geotransolver", case_id=None):
    """Color every original vertex by interval containment; no color-scale comparison is needed."""
    from IPython.display import Image, display
    import pyvista as pv
    result = load_uq_results()
    case_id = str(case_id or result["validation_evaluation_case_ids"][0])
    data = _load_case(model, case_id, result)
    error = np.abs(data["target"][:, 0]-data["mean"][:, 0])
    std = data["std"][:, 0]
    q = result["models"][model]["calibration"]["pressure_q"]
    calibrated = q*np.maximum(std, 1e-8) if model == "geotransolver" else np.full_like(error, q)
    intervals = [("Raw MC band", 1.645*std), ("Calibrated 90% interval", calibrated)] if model == "geotransolver" else [("Calibrated residual interval", calibrated)]
    masks = [(error <= half).astype(np.int8) for _, half in intervals]
    plotter = pv.Plotter(shape=(1, len(intervals)), off_screen=True,
                         window_size=(900*len(intervals), 630))
    plotter.set_background("white")
    for index, ((label, half), mask) in enumerate(zip(intervals, masks)):
        plotter.subplot(0, index)
        mesh = pv.PolyData(np.array(data["coordinates"], copy=True))
        name = f"contains_reference_panel_{index}"
        mesh.point_data[name] = np.array(mask, copy=True)
        plotter.add_mesh(mesh, scalars=name, style="points", point_size=3,
                         cmap=["#cc3311", "#228833"], clim=(0, 1),
                         show_scalar_bar=False, render_points_as_spheres=False,
                         ambient=1., diffuse=0.)
        plotter.add_text(f"{label}\nCase {case_id}; {mask.mean():.1%} vertices contained",
                         font_size=12, position="upper_left", color="black")
        plotter.add_text("Green: contains CFD pressure\nRed: misses CFD pressure\nEach rendered point is one original vertex",
                         font_size=10, position="lower_left", color="black")
        plotter.camera_position = "iso"; plotter.camera.zoom(1.03)
    plotter.link_views()
    path = _figure_cache()/f"coverage_{model}_case{case_id}.png"
    plotter.screenshot(str(path)); plotter.close()
    proof = dict(case_id=case_id, model=model, checkpoint_sha256=result["models"][model]["checkpoint_sha256"],
                 pressure_q=q, vertex_count=len(error), interval_labels=[label for label, _ in intervals],
                 vertex_coverage=[float(mask.mean()) for mask in masks],
                 mask_sha256=[hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest() for mask in masks],
                 png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 association="Original vertices, rendered as points; no interpolation or downsampling",
                 result_source=_result_note(result))
    path.with_suffix(".json").write_text(json.dumps(proof, indent=2)+"\n")
    display(Image(filename=str(path)))


__all__ = ['binary_stl_summary', 'build_geometry_reference', 'validate_geometry_reference', 'load_geometry_reference', 'geometry_scale', 'geometry_features', 'normalize_geometry', 'velocity_conditioning', 'locate_geometry_reference', 'plot_body_fields', 'plot_geometry_comparison', 'plot_field_distributions', 'plot_attention_comparison', 'plot_pressure_snapshot', 'plot_spatial_statistics', 'plot_spatial_clusters', 'plot_slice_assignments', 'plot_token_features', 'plot_token_attention', 'configure_tutorial_style', 'plot_geometry_cloud', 'plot_coordinate_snapshots', 'plot_coordinate_drift', 'plot_context_sensitivity', 'plot_gate_coefficients', 'plot_surface_data_contract', 'render_mesh_panels', 'find_ahmed_vtp', 'original_surface_mesh', 'export_prediction_vtp', 'plot_surface_predictions', 'render_surface_comparison', 'plot_measured_training', 'capture_slice_weights', 'plot_slice_memberships', 'plot_routing_entropy', 'plot_sample_field_parity', 'VelocityOnlyDataPipe', 'load_velocity_notebook_support', 'restore_native_fields', 'resolve_case_flow', 'integrate_surface_traction', 'force_coefficient_components', 'regression_metrics', 'plot_metadata_curation_audit', 'load_normalization', 'FixedCaseDataPipe', 'field_metrics', 'plot_geo_input_flow', 'show_uq_result_source', 'load_uq_results', 'show_uq_drag', 'show_uq_summary', 'show_uq_figures', 'show_uq_figure', 'plot_uq_workflow', 'plot_toy_predictions', 'plot_toy_calibration', 'show_training_evidence', 'run_actual_training_demo', 'show_uq_surface_examples', 'show_uq_pipeline', 'show_uq_drag_intervals', 'show_uq_pointwise_diagnostics', 'show_uq_force_workflow', 'show_training_interpretation', 'show_uq_interpretation', 'show_uq_coverage_surface']


# ---------------------------------------------------------------------------
# Public tutorial execution and verified recorded outputs
# ---------------------------------------------------------------------------







# ===========================================================================
# Executed checkpoint and five-epoch training lessons
# ===========================================================================

"""Readable checkpoint and five-epoch lessons for the surface-model notebooks.

The public functions are appended to the notebooks' single ``utils.py``.  The
matching external experiment backend owns data preparation and training; no
scientific transform or optimization loop is reimplemented here.
"""


def _lesson_support():
    import importlib.util
    import os
    from pathlib import Path

    configured = os.environ.get("VELOCITY_NOTEBOOK_SUPPORT")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        root = os.environ.get("FULL_DATA_UQ_ROOT")
        if not root:
            raise RuntimeError("Set FULL_DATA_UQ_ROOT to the matching external experiment backend before running with data.")
        path = Path(root).expanduser().resolve() / "notebook_support.py"
    if not path.is_file():
        raise FileNotFoundError("The external notebook backend is missing: " + str(path))
    spec = importlib.util.spec_from_file_location("_five_epoch_lesson_backend", path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    if not hasattr(support, "training_directory") or not hasattr(support, "TARGET_EPOCHS"):
        raise ValueError("Use the final-checkpoint backend supporting Transolver 600 and GeoTransolver 500.")
    return support


def _lesson_source(model_type, source_results):
    from pathlib import Path

    if model_type not in ("transolver", "geotransolver"):
        raise ValueError("Choose transolver or geotransolver.")
    support = _lesson_support()
    source = Path(support._host_path(source_results)).resolve()
    if source != Path(support.training_directory(model_type)).resolve():
        raise ValueError("Start this lesson from the declared final production training directory.")
    checkpoint, manifest, settings = support._validate_checkpoint(source / "last_model.pt")
    if settings["model_type"] != model_type or int(checkpoint["epoch"]) != support.TARGET_EPOCHS[model_type]:
        raise ValueError("The model family or completed checkpoint epoch does not match the lesson.")
    return support, source, checkpoint, manifest, settings


def _lesson_hash(path):
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_training_checkpoint(model_type, source_results):
    """Load the real final model and explain one genuine validation batch.

    This performs a strict model-state load and a forward pass. It reads saved
    optimizer/scheduler metadata without changing the checkpoint or its weights.
    Returned values are plain numbers and lists suitable for notebook outputs.
    """
    import gc
    import torch

    support, source, checkpoint, manifest, settings = _lesson_source(model_type, source_results)
    restored = support.load_notebook_case(source / "last_model.pt")
    model, batch = restored["model"], restored["batch"]
    with torch.no_grad():
        if model_type == "geotransolver":
            prediction = model(global_embedding=batch["fx"], local_embedding=batch["embeddings"],
                               geometry=batch["geometry"], local_positions=batch["embeddings"][..., :3])
        else:
            prediction = model(fx=batch["fx"], embedding=batch["embeddings"])
        sample_mse = float(torch.nn.functional.mse_loss(prediction, batch["fields"]))
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    children = [{"name": name, "module_type": type(module).__name__,
                 "parameters": sum(value.numel() for value in module.parameters())}
                for name, module in model.named_children()]
    counts = {role: len(checkpoint[role + "_case_ids"]) for role in ("train", "validation", "calibration")}
    adam_steps = sorted({int(float(value["step"])) for value in checkpoint["optimizer"]["state"].values()})
    scheduler = checkpoint["scheduler"]
    info = {
        "model_type": model_type, "model_class": type(model).__name__,
        "checkpoint_path": str(source / "last_model.pt"), "checkpoint_sha256": _lesson_hash(source / "last_model.pt"),
        "source_epoch": int(checkpoint["epoch"]), "strict_model_load": True,
        "total_parameters": parameters, "trainable_parameters": trainable,
        "parameter_storage_mib": sum(value.numel() * value.element_size() for value in model.parameters()) / 1024**2,
        "parameter_groups": children, "model_config": dict(settings["model_config"]),
        "batch_shapes": {name: list(value.shape) for name, value in batch.items()},
        "batch_case_id": str(restored["case_id"]), "batch_role": "validation",
        "inlet_velocity": float(batch["fx"].reshape(-1)[0]),
        "first_local_features": batch["embeddings"][0, 0].detach().cpu().tolist(),
        "first_normalized_target": batch["fields"][0, 0].detach().cpu().tolist(),
        "first_normalized_prediction": prediction[0, 0].detach().cpu().tolist(),
        "prediction_shape": list(prediction.shape), "sample_normalized_mse": sample_mse,
        "target_channels": ["pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z"],
        "target_mean": checkpoint["normalization"]["mean"], "target_std": checkpoint["normalization"]["std"],
        "split_counts": counts, "batch_size": 1, "optimizer_steps_per_epoch": counts["train"],
        "optimizer": "AdamW", "optimizer_state_entries": len(checkpoint["optimizer"]["state"]),
        "completed_adam_steps": adam_steps, "learning_rates": [float(group["lr"]) for group in checkpoint["optimizer"]["param_groups"]],
        "scheduler": "update-based warmup/cosine" if model_type == "geotransolver" else "StepLR",
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_completed_updates": scheduler.get("completed_updates"),
        "gradient_clip": float(settings["gradient_clip"]), "weight_decay": float(settings["weight_decay"]),
        "regularization_weight": float(settings.get("lambda_reg", 0.0)),
        "device": str(next(model.parameters()).device), "gpu": torch.cuda.get_device_name(),
        "dtype": str(next(model.parameters()).dtype),
        "weights_changed": False,
    }
    title = "GeoTransolver" if model_type == "geotransolver" else "Transolver"
    print(f"{title}: loaded last_model.pt at completed epoch {info['source_epoch']} (strict state match).")
    print(f"Parameters: {parameters:,} total; {trainable:,} trainable; {info['parameter_storage_mib']:.1f} MiB for parameter tensors.")
    print(f"Architecture: {settings['model_config']['n_layers']} layers, hidden width {settings['model_config']['n_hidden']}, "
          f"{settings['model_config']['n_head']} heads, {settings['model_config']['slice_num']} slices per head.")
    print("Parameter groups:")
    for child in children:
        print(f"  {child['name']:<24} {child['parameters']:>12,}  ({child['module_type']})")
    print(f"One real {info['batch_role']} body: case {info['batch_case_id']}; inlet velocity {info['inlet_velocity']:.6g}.")
    for name, shape in info["batch_shapes"].items():
        print(f"  {name:<12} {tuple(shape)}")
    print("  prediction  " + str(tuple(info["prediction_shape"])))
    print("First sampled location [normalized x, y, z, normal x, y, z]:")
    print("  " + ", ".join(f"{value:.6f}" for value in info["first_local_features"]))
    print("First normalized target [pressure, shear x, shear y, shear z]:")
    print("  " + ", ".join(f"{value:.6f}" for value in info["first_normalized_target"]))
    print("First normalized model prediction in the same channel order:")
    print("  " + ", ".join(f"{value:.6f}" for value in info["first_normalized_prediction"]))
    print(f"This single validation body's normalized four-channel MSE: {sample_mse:.6f}.")
    print(f"One epoch: {counts['train']} training bodies = {counts['train']} optimizer steps; then {counts['validation']} validation bodies.")
    print(f"Saved AdamW state: {info['optimizer_state_entries']} parameter tensors; completed step counter(s) {adam_steps}.")
    print(f"Resume learning rate: {info['learning_rates'][0]:.8g}; scheduler: {info['scheduler']}; gradient clip: {info['gradient_clip']:.6g}.")
    print("The forward pass checks loading and shapes. The five-epoch continuation below restores the saved optimizer, scheduler, and random state.")
    del prediction, model, batch, restored, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    globals().setdefault("_TRAINING_LESSON_INSPECTIONS", {})[(model_type, str(source))] = info
    return info


def _lesson_validate_output(support, source_checkpoint, output_dir, model_type, source_epoch):
    import json
    import torch

    checkpoint = torch.load(output_dir / "last_model.pt", map_location="cpu", weights_only=False)
    history = json.loads((output_dir / "learning_curve.json").read_text())
    if (int(checkpoint["epoch"]) != source_epoch + 5 or checkpoint["history"] != history
            or history[:source_epoch] != source_checkpoint["history"]
            or [row["epoch"] for row in history] != list(range(1, source_epoch + 6))):
        raise ValueError("The five actual epoch records do not match the completed output checkpoint.")
    trainer = support._trainer(model_type)
    settings = json.loads((output_dir / "settings.json").read_text())
    trainer.checkpoint_compatible(checkpoint, {"settings": settings}, source_checkpoint["experiment_signature"])
    if checkpoint["normalization"] != source_checkpoint["normalization"]:
        raise ValueError("The continuation changed target normalization.")
    expected_steps = (source_epoch + 5) * len(checkpoint["train_case_ids"])
    if any(int(float(value["step"])) != expected_steps for value in checkpoint["optimizer"]["state"].values()):
        raise ValueError("Saved AdamW counters do not include all five training epochs.")
    rows = history[source_epoch:]
    if any(row["train_cases"] != 458 or row["validation_cases"] != 30 or row["first_step_parameter_max_abs_update"] <= 0 for row in rows):
        raise ValueError("The lesson must use all training and validation cases and perform real parameter updates.")
    return checkpoint, rows


def run_five_epoch_lesson(model_type, source_results, output_dir):
    """Run five real epochs from the final checkpoint in a separate directory.

    Completed output is reused only after checking its source hash, checkpoint,
    full history, and optimizer counters. Training still uses the frozen external
    backend, including its dynamic subsampling and saved random-state restore.
    """
    import gc
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    support, source, checkpoint, manifest, settings = _lesson_source(model_type, source_results)
    out = Path(output_dir).expanduser().resolve()
    demo_root = (Path(support.ROOT) / "notebook_demos").resolve()
    if demo_root not in out.parents or source == out or source in out.parents or out in source.parents:
        raise ValueError("Use a separate output directory inside the configured backend's notebook_demos directory.")
    start = int(checkpoint["epoch"])
    source_hash = _lesson_hash(source / "last_model.pt")
    inspections = globals().get("_TRAINING_LESSON_INSPECTIONS", {})
    info = inspections.get((model_type, str(source)))
    if info is None or info.get("checkpoint_sha256") != source_hash:
        info = inspect_training_checkpoint(model_type, source)
    before = support._snapshot_production()
    summary_path = out / "lesson_summary.json"
    cached = summary_path.is_file()
    try:
        if cached:
            summary = json.loads(summary_path.read_text())
            if (summary.get("status") != "complete" or summary.get("model_type") != model_type
                    or summary.get("source_checkpoint_sha256") != source_hash
                    or summary.get("additional_epochs") != 5
                    or summary.get("output_checkpoint_sha256") != _lesson_hash(out / "last_model.pt")):
                raise ValueError("Existing lesson output belongs to a different source or was changed; use a new directory.")
            final, rows = _lesson_validate_output(support, checkpoint, out, model_type, start)
            progress = summary["batch_progress"]
            print(f"VERIFIED RECORDED OUTPUT: five actual epochs {start + 1}–{start + 5}; this rerun performs no new updates.")
        else:
            if out.exists() and any(out.iterdir()):
                raise ValueError("The five-epoch lesson requires a new empty output directory.")
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "best_model.pt", out / "best_model.pt")
            demo_settings = dict(settings, epochs=start + 5, geometry_reference=str(source / "geometry_reference.json"))
            config = out / "lesson_config.json"
            config.write_text(json.dumps(demo_settings, indent=2, allow_nan=False) + "\n")
            trainer = support._trainer(model_type)
            command = [sys.executable, "-u", str(Path(trainer.__file__).resolve()), "--config", str(config),
                       "--output-dir", str(out), "--resume", str(source / "last_model.pt"), "--segment-epochs", "5"]
            print(f"Restoring model weights, AdamW moments, scheduler, and RNG from epoch {start}.")
            print(f"Running five real epochs: {start + 1}–{start + 5}; 458 training steps and 30 validation cases per epoch.")
            print("Training points are resampled; the validation sample is fixed for each body.")
            progress = []
            with (out / "training_stdout.log").open("w", buffering=1) as raw_log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        raw_log.write(line)
                        try:
                            record = json.loads(line)
                        except (ValueError, TypeError):
                            print(line.rstrip(), flush=True)
                            continue
                        stage = record.get("stage")
                        if stage == "train":
                            progress.append(record)
                            print(f"Epoch {record['epoch']} | batch {record['step'] + 1:3d}/{record['total']} | "
                                  f"current batch MSE {record['normalized_mse']:.6f}", flush=True)
                        elif stage == "epoch_complete":
                            print(f"Epoch {record['epoch']} complete | train {record['train_mse']:.6f} | "
                                  f"validation {record['validation_mse']:.6f}", flush=True)
                        elif stage == "training_only_normalization":
                            print(f"Reading training targets for normalization: {record['cases']}/{record['total']} bodies.", flush=True)
                        elif stage in ("complete", "training_complete", "resume"):
                            print(f"Trainer: {stage}.", flush=True)
                    if process.wait() != 0:
                        raise RuntimeError("Five-epoch training failed; inspect training_stdout.log in the lesson output directory.")
            final, rows = _lesson_validate_output(support, checkpoint, out, model_type, start)
        rates = [float(group["lr"]) for group in checkpoint["optimizer"]["param_groups"]]
        epoch_rows = []
        for row in rows:
            epoch_rows.append(dict(row, learning_rate_used=list(rates)))
            rates = row["learning_rates"]
        summary = dict(status="complete", model_type=model_type, source_epoch=start, final_epoch=start + 5,
                       additional_epochs=5, source_checkpoint_sha256=source_hash,
                       output_checkpoint_sha256=_lesson_hash(out / "last_model.pt"),
                       checkpoint_path=str(out / "last_model.pt"), output_dir=str(out),
                       load_inspection=info, epochs=epoch_rows, batch_progress=progress,
                       source_artifacts_unchanged=True, production_artifacts_unchanged=True,
                       evaluated_production_epoch=start, checkpoint_selection_policy="last_completed_epoch",
                       teaching_checkpoint_used_for_reported_metrics=False)
    finally:
        if support._snapshot_production() != before:
            raise RuntimeError("A protected production artifact changed during the isolated lesson.")
    if not cached:
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, summary_path)
    print("\nEpoch  Train MSE  Validation MSE  Learning rate used  Seconds")
    for row in epoch_rows:
        print(f"{row['epoch']:5d}  {row['train_mse']:.6f}   {row['validation_mse']:.6f}       "
              f"{row['learning_rate_used'][0]:.8g}       {row['seconds']:.1f}")
    print(f"Saved lesson checkpoint: epoch {start + 5}.")
    print(f"Verified actual optimizer updates: {5 * info['optimizer_steps_per_epoch']:,} across five epochs.")
    print(f"The reported accuracy and UQ results still use the original epoch-{start} production checkpoint.")
    print("Production weights, histories, normalization, and evaluation files are unchanged.")
    summary["recorded_output"] = cached
    del checkpoint, final
    gc.collect()
    return summary


def plot_training_lesson(lesson):
    """Compare five measured epochs on the same zero-based MSE scale."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = lesson["epochs"]
    if len(rows) != 5:
        raise ValueError("This plot requires five measured epoch records.")
    epochs = [row["epoch"] for row in rows]
    title = "GeoTransolver" if lesson["model_type"] == "geotransolver" else "Transolver"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    upper = 1.15 * max(float(row[key]) for row in rows for key in ("train_mse", "validation_mse"))
    for ax, key, label, color in zip(axes, ("train_mse", "validation_mse"),
                                   ("Training: 458 bodies", "Validation: 30 bodies"), ("#2878a5", "#d27a36")):
        values = [row[key] for row in rows]
        ax.plot(epochs, values, "o-", color=color, linewidth=2, markersize=6)
        ax.set(xlabel="Completed epoch", ylabel="Normalized four-channel MSE", title=label)
        ax.set_xticks(epochs)
        ax.set_ylim(0, upper)
        ax.annotate(f"{values[-1]:.5f}", (epochs[-1], values[-1]), xytext=(-8, 10),
                    textcoords="offset points", ha="right", fontsize=9, color=color)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=.2)
    fig.suptitle(f"{title}: five actual epochs after loading epoch {lesson['source_epoch']}", fontsize=13)
    return fig


__all__ += ['inspect_training_checkpoint', 'run_five_epoch_lesson', 'plot_training_lesson']


# ===========================================================================
# Readable model architecture summaries
# ===========================================================================

"""Small, notebook-friendly architecture summaries from an instantiated model.

No forward pass is performed.  Counts come from the model's actual parameter
objects, with shared parameter objects counted once in each displayed subtree.
"""


def format_model_summary(
    model, *, model_label=None, max_depth=2, expand_paths=("blocks.0",)
):
    """Return an ASCII module tree and exact parameter counts.

    ``max_depth`` counts the root as depth zero.  ``expand_paths`` additionally
    exposes the immediate children of the named modules, making the first
    repeated block readable without expanding every repeated block.  A row's
    count includes its descendants; the final total deduplicates all shared
    parameter objects across the complete model.
    """
    import textwrap

    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if isinstance(expand_paths, str):
        raise TypeError("expand_paths must contain module paths, not one string")
    expand_paths = tuple(str(path) for path in expand_paths)

    def ascii_text(value):
        return str(value).encode("ascii", "backslashreplace").decode("ascii")

    def unique_parameters(module):
        return {id(parameter): parameter for parameter in module.parameters()}

    def parameter_count(module):
        return sum(parameter.numel() for parameter in unique_parameters(module).values())

    def expanded(path, depth):
        return depth < max_depth or (not path and bool(expand_paths)) or any(
            requested == path or requested.startswith(path + ".")
            for requested in expand_paths
        )

    name_width = 76
    number_width = 21
    line_width = name_width + 2 + number_width
    rule = "=" * line_width

    def row(name, count):
        name = ascii_text(name)
        if len(name) > name_width:
            name = name[: name_width - 3] + "..."
        number = f"{count:,}" if count else "--"
        return f"{name:<{name_width}}  {number:>{number_width}}"

    lines = ["[Model summary]"]
    if model_label:
        lines.extend(textwrap.wrap(ascii_text(model_label), width=line_width))
    lines.extend(
        (
            rule,
            f"{'Module (name: type)':<{name_width}}  {'Param # (subtree)':>{number_width}}",
            rule,
        )
    )
    lines.append(row(type(model).__name__, parameter_count(model)))

    def visit(module, path, depth, prefix):
        if not expanded(path, depth):
            return
        children = list(module.named_children())
        for index, (name, child) in enumerate(children):
            last = index == len(children) - 1
            branch = "+-- " if last else "|-- "
            child_path = f"{path}.{name}" if path else name
            lines.append(
                row(prefix + branch + f"{name}: {type(child).__name__}", parameter_count(child))
            )
            visit(child, child_path, depth + 1, prefix + ("    " if last else "|   "))

    visit(model, "", 0, "")
    parameters = unique_parameters(model)
    total = sum(parameter.numel() for parameter in parameters.values())
    trainable = sum(
        parameter.numel() for parameter in parameters.values() if parameter.requires_grad
    )
    storage_mib = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters.values()
    ) / (1024 ** 2)
    lines.extend(
        (
            rule,
            f"Total parameters:          {total:,}",
            f"Trainable parameters:      {trainable:,}",
            f"Non-trainable parameters:  {total - trainable:,}",
            f"Parameter tensor storage:  {storage_mib:.2f} MiB",
            rule,
            "Counts include each module's children; do not add parent and child rows.",
            "The total counts every shared parameter object once. -- means zero parameters.",
            "Storage covers parameter tensors only; no forward pass or training-memory estimate.",
        )
    )
    return "\n".join(lines)


def print_model_summary(model, *, model_label=None, max_depth=2, expand_paths=("blocks.0",)):
    """Print the architecture tree once, without a duplicate notebook return value."""
    print(
        format_model_summary(
            model,
            model_label=model_label,
            max_depth=max_depth,
            expand_paths=expand_paths,
        )
    )


__all__ += ['format_model_summary', 'print_model_summary']


# Captured from strictly loaded final checkpoints; no data access is needed to read these trees.
_RECORDED_MODEL_SUMMARIES = {
    'transolver': """[Model summary]
Recorded Transolver architecture | checkpoint epoch 600 | strict load passed
===================================================================================================
Module (name: type)                                                               Param # (subtree)
===================================================================================================
Transolver                                                                                3,987,268
|-- preprocess: _TransolverMlp                                                              135,424
|   +-- layers: Sequential                                                                  135,424
+-- blocks: ModuleList                                                                    3,851,844
    |-- 0: TransolverBlock                                                                  481,288
    |   |-- ln_1: LayerNorm                                                                     512
    |   |-- Attn: PhysicsAttentionIrregularMesh                                             217,352
    |   +-- ln_mlp1: Sequential                                                             263,424
    |-- 1: TransolverBlock                                                                  481,288
    |-- 2: TransolverBlock                                                                  481,288
    |-- 3: TransolverBlock                                                                  481,288
    |-- 4: TransolverBlock                                                                  481,288
    |-- 5: TransolverBlock                                                                  481,288
    |-- 6: TransolverBlock                                                                  481,288
    +-- 7: TransolverBlock                                                                  482,828
===================================================================================================
Total parameters:          3,987,268
Trainable parameters:      3,987,268
Non-trainable parameters:  0
Parameter tensor storage:  15.21 MiB
===================================================================================================
Counts include each module's children; do not add parent and child rows.
The total counts every shared parameter object once. -- means zero parameters.
Storage covers parameter tensors only; no forward pass or training-memory estimate.
""",
    'geotransolver': """[Model summary]
Recorded GeoTransolver architecture | checkpoint epoch 500 | strict load passed
===================================================================================================
Module (name: type)                                                               Param # (subtree)
===================================================================================================
GeoTransolver                                                                            21,660,586
|-- context_builder: GlobalContextBuilder                                                   121,206
|   |-- local_extractors: ModuleList                                                        109,668
|   |-- geometry_tokenizer: ContextProjector                                                  6,281
|   +-- global_tokenizer: ContextProjector                                                    5,257
|-- preprocess: ModuleList                                                                  134,912
|   +-- 0: _TransolverMlp                                                                   134,912
|-- blocks: ModuleList                                                                   21,402,160
|   |-- 0: GALEBlock                                                                      1,070,108
|   |   |-- ln_1: LayerNorm                                                                     768
|   |   |-- Attn: GALE                                                                      477,594
|   |   |-- ln_mlp1: Sequential                                                             591,744
|   |   |-- attn_dropout: ConcreteDropout                                                         1
|   |   +-- ffn_dropout: ConcreteDropout                                                          1
|   |-- 1: GALEBlock                                                                      1,070,108
|   |-- 2: GALEBlock                                                                      1,070,108
|   |-- 3: GALEBlock                                                                      1,070,108
|   |-- 4: GALEBlock                                                                      1,070,108
|   |-- 5: GALEBlock                                                                      1,070,108
|   |-- 6: GALEBlock                                                                      1,070,108
|   |-- 7: GALEBlock                                                                      1,070,108
|   |-- 8: GALEBlock                                                                      1,070,108
|   |-- 9: GALEBlock                                                                      1,070,108
|   |-- 10: GALEBlock                                                                     1,070,108
|   |-- 11: GALEBlock                                                                     1,070,108
|   |-- 12: GALEBlock                                                                     1,070,108
|   |-- 13: GALEBlock                                                                     1,070,108
|   |-- 14: GALEBlock                                                                     1,070,108
|   |-- 15: GALEBlock                                                                     1,070,108
|   |-- 16: GALEBlock                                                                     1,070,108
|   |-- 17: GALEBlock                                                                     1,070,108
|   |-- 18: GALEBlock                                                                     1,070,108
|   +-- 19: GALEBlock                                                                     1,070,108
+-- ln_mlp_out: ModuleList                                                                    2,308
    +-- 0: Sequential                                                                         2,308
===================================================================================================
Total parameters:          21,660,586
Trainable parameters:      21,660,586
Non-trainable parameters:  0
Parameter tensor storage:  82.63 MiB
===================================================================================================
Counts include each module's children; do not add parent and child rows.
The total counts every shared parameter object once. -- means zero parameters.
Storage covers parameter tensors only; no forward pass or training-memory estimate.
""",
}

def show_saved_model_summary(model_type):
    """Display the actual saved architecture tree without loading model data.

    These records describe the production checkpoints used by the lessons:
    Transolver epoch 600 and GeoTransolver epoch 500. Use print_model_summary
    with a loaded model to inspect a different architecture.
    """
    if model_type not in _RECORDED_MODEL_SUMMARIES:
        raise ValueError("Choose 'transolver' or 'geotransolver'.")
    print(_RECORDED_MODEL_SUMMARIES[model_type], end="")


__all__ += ['show_saved_model_summary']
