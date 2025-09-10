import numpy as np
import vtk
import os
import re
import torch
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import torch
import apex
from tqdm import tqdm
from torch.cuda.amp import autocast
from scipy.interpolate import griddata
from vtk.util import numpy_support
from scipy.spatial import KDTree
from pathlib import Path
from physicsnemo.launch.utils import save_checkpoint
from physicsnemo.utils.sdf import signed_distance_field
from physicsnemo.utils.domino.utils import *
from physicsnemo.distributed import DistributedManager

vtk.vtkObject.GlobalWarningDisplayOff()
os.environ["NO_AT_BRIDGE"] = "1"
pv.start_xvfb()


def normalize_to_box(coords, min_val, max_val):
    """Normalizes coordinates to the bounding box [-1, 1]."""
    return 2.0 * (coords - min_val) / (max_val - min_val) - 1.0

def unnormalize(data, min_val, max_val):
    """Unnormalizes data from a normalized range back to its original scale."""
    return data * (max_val - min_val) + min_val

def dict_to_device(data_dict, device):
    """Moves all tensors in a dictionary to the specified device (e.g., 'cuda' or 'cpu')."""
    return {k: v.to(device) for k, v in data_dict.items()}

def calculate_center_of_mass(points, masses):
    """Calculates the center of mass for a set of points with given masses."""
    return np.sum(points * masses[:, np.newaxis], axis=0) / np.sum(masses)


def get_test_case_info(vtp_file_path):
    """Extracts the tag from 'case*.vtp*' directory and returns the STL file path inside it."""

    vtp_file_path = Path(vtp_file_path)
    # Extract the dataset root automatically
    dataset_root = vtp_file_path.parent.parent  # goes from .../dataset/test/ → .../dataset


    # Extract the tag from the VTP filename
    match = re.search(r"case(\d+)", vtp_file_path.stem)
    if not match:
        raise ValueError(f"Could not extract tag from VTP file name: {vtp_file_path.name}")
    tag = int(match.group(1))

    # Construct the corresponding STL path
    stl_path = dataset_root / "test_stl_files" / f"case{tag}.stl"
    if not stl_path.exists():
        raise FileNotFoundError(f"No STL file found for tag {tag} in {stl_path.parent}")
    return tag, stl_path



# --- Data Pre-processing ---

def process_stl_file(stl_path):
    """Reads and processes a .stl file to extract geometric properties."""
    mesh_stl = pv.read(stl_path)
    vertices = mesh_stl.points.astype(np.float32)
    faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
    
    length_scale = np.amax(np.amax(vertices, 0) - np.amin(vertices, 0))
    sizes = np.array(mesh_stl.compute_cell_sizes(area=True).cell_data["Area"], dtype=np.float32)
    centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)
    
    return {
        "vertices": vertices,
        "faces_indices": faces.flatten(),
        "length_scale": length_scale,
        "center_of_mass": calculate_center_of_mass(centers, sizes),
    }

def create_sdf_grid(vertices, faces_indices, grid_resolution, bounding_box):
    """Creates a grid and computes the Signed Distance Field (SDF)."""
    nx, ny, nz = grid_resolution
    s_max = np.float32(bounding_box.max)
    s_min = np.float32(bounding_box.min)

    # Create a uniform grid
    grid = np.mgrid[s_min[0]:s_max[0]:nx*1j, s_min[1]:s_max[1]:ny*1j, s_min[2]:s_max[2]:nz*1j].transpose(1, 2, 3, 0)
    grid_reshaped = grid.reshape(-1, 3)

    # This assumes you have a 'signed_distance_field' function available
    sdf_grid = signed_distance_field(
        vertices, faces_indices, grid_reshaped
    ).reshape(nx, ny, nz)

    return {
        "grid": grid.astype(np.float32),
        "sdf": sdf_grid.astype(np.float32),
        "grid_min_max": np.float32([s_min, s_max]),
    }

def process_vtp_data(vtp_file, num_neighbors, surface_variable_names, bounding_box):
    """Reads and processes a .vtp file to extract surface mesh data."""

    s_max = np.float32(bounding_box.max)
    s_min = np.float32(bounding_box.min)

#    vtp_file = Path(vtp_path) / "boundary.vtp"
    mesh = pv.read(vtp_file)

    # Read the VTP file containing surface data
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    polydata_surf = reader.GetOutput()
    celldata_all = get_node_to_elem(polydata_surf)
    celldata = celldata_all.GetCellData()
    surface_fields = get_fields(celldata, surface_variable_names)
    surface_fields = np.concatenate(surface_fields, axis=-1)
    mesh = pv.PolyData(polydata_surf)


    # Extract surface mesh coordinates, neighbors, and normals
    surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    interp_func = KDTree(surface_coordinates)
    dd, ii = interp_func.query(surface_coordinates, k=num_neighbors)
    surface_neighbors = surface_coordinates[ii]
    surface_neighbors = surface_neighbors[:, 1:]
    surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_sizes = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)

     
    # Normalize the surface normals and neighbors
    surface_normals = (
        surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
    )
    surface_neighbors_normals = surface_normals[ii]
    surface_neighbors_normals = surface_neighbors_normals[:, 1:]
    surface_neighbors_sizes = surface_sizes[ii]
    surface_neighbors_sizes = surface_neighbors_sizes[:, 1:]

    surface_coordinates = normalize(surface_coordinates, s_max, s_min)
    surface_neighbors = normalize(surface_neighbors, s_max, s_min)
  

    return {
        "polydata": mesh,
        "fields": surface_fields,
        "coordinates": surface_coordinates,
        "neighbors": surface_neighbors,
        "normals": surface_normals,
        "neighbors_normals": surface_neighbors_normals,
        "sizes": surface_sizes,
        "neighbors_sizes": surface_neighbors_sizes,
    }

def assemble_data_dict(stl_data, sdf_data, vtp_data, stream_velocity, air_density):
    """Assembles the final data dictionary and converts numpy arrays to PyTorch tensors."""
    s_max = sdf_data["grid_min_max"][1]
    s_min = sdf_data["grid_min_max"][0]
    
    # Helper for normalization
    def normalize(data): return (data - s_min) / (s_max - s_min)

    data_dict_np = {
        "pos_surface_center_of_mass": vtp_data["coordinates"] - stl_data["center_of_mass"],
        "geometry_coordinates": stl_data["vertices"],
        "surf_grid": normalize(sdf_data["grid"]),
        "sdf_surf_grid": sdf_data["sdf"],
        "surface_mesh_centers": normalize(vtp_data["coordinates"]),
        "surface_mesh_neighbors": normalize(vtp_data["neighbors"]),
        "surface_normals": vtp_data["normals"],
        "surface_neighbors_normals": vtp_data["neighbors_normals"],
        "surface_areas": vtp_data["sizes"],
        "surface_neighbors_areas": vtp_data["neighbors_sizes"],
        "surface_fields": vtp_data["fields"],
        "surface_min_max": sdf_data["grid_min_max"],
        "length_scale": stl_data["length_scale"],
        "stream_velocity": np.expand_dims(np.array(stream_velocity, dtype=np.float32), axis=-1),
        "air_density": np.expand_dims(np.array(air_density, dtype=np.float32), axis=-1),
    }

    # Convert all numpy arrays to tensors with a batch dimension
    return {
        k: torch.from_numpy(np.expand_dims(v.astype(np.float32), 0))
        for k, v in data_dict_np.items()
    }


# --- Post-processing and Saving ---

def unscale_surface_prediction(prediction_tensor, factors, velocity, density):
    """Unnormalizes and scales the model's surface prediction to physical values."""
    pred_np = prediction_tensor.cpu().numpy()
    unnormalized_pred = unnormalize(pred_np, factors[0], factors[1])
    return unnormalized_pred * velocity**2.0 * density

def calculate_forces(prediction, truth, normals, areas):
    """Calculates and returns the predicted and true forces along the x-axis."""
    areas = np.expand_dims(areas, -1) # Ensure correct shape for broadcasting
    
    force_x_pred = np.sum(
        prediction[0, :, 0] * normals[:, 0] * areas[:, 0] -
        prediction[0, :, 1] * areas[:, 0]
    )
    
    force_x_true = np.sum(
        truth[:, 0] * normals[:, 0] * areas[:, 0] -
        truth[:, 1] * areas[:, 0]
    )
    
    return force_x_pred, force_x_true

def save_predictions_to_vtp(polydata, prediction, var_names, output_path):
    """Attaches prediction arrays to polydata and saves to a new VTP file."""
    for i, name in enumerate(var_names):
        polydata.cell_data[f"{name}Pred"] = prediction[0, :, i]

    polydata.save(output_path)
    print(f"Predictions saved to {output_path}")

